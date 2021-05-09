import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
#from utils import resnet
from utils.freihandnet import MyPoseHand, HM2Mano, normal_init#, mesh2poseNet
from utils.net_hg import Net_HM_HG
#from utils.hand_det import handpose_model

import time

import util
from util import face_vertices, json_load

from functools import partial
from six.moves import map, zip


from torch.autograd import Variable
#from efficientnet_pytorch import EfficientNet
try:
    from efficientnet_pt.model import EfficientNet
except:
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from efficientnet_pt import EfficientNet

#from utils.fh_utils import AverageMeter

# encoder efficientnet
class Encoder(nn.Module):
    def __init__(self,version='b3'):
        super(Encoder, self).__init__()
        self.version = version
        if self.version == 'b3':
            #self.encoder = EfficientNet.from_pretrained('efficientnet-b3')
            self.encoder = EfficientNet.from_name('efficientnet-b3')
            # b3 [1536,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)
        '''
        elif self.version == 'b5':
            self.encoder = EfficientNet.from_pretrained('efficientnet-b5')
            # b5 [2048,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)
        '''
    def forward(self, x):
        features, low_features = self.encoder.extract_features(x)#[B,1536,7,7] [B,32,56,56]
        features = self.pool(features)
        features = features.view(features.shape[0],-1)##[B,1536]
        return features, low_features

'''
class Percep_Encoder(nn.Module):
    def __init__(self):
        super(Percep_Encoder, self).__init__()
        self.percep_encoder = EfficientNet.from_pretrained('efficientnet-b0')
    def forward(self, x):
        y = self.percep_encoder(x)
        return y
'''

def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    #return ((im.float() / 255.0) - 0.5)
    '''
    :param im: torch byte tensor, B x C x H x W, 0 ~ 1
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    '''
    return (im - 0.5)

class RGB2HM(nn.Module):
    def __init__(self):
        super(RGB2HM, self).__init__()
        num_joints = 21
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=2,
                                num_modules=2,
                                num_feats=256)
    def forward(self, images):
        images = normalize_image(images)
        # 1. Heat-map estimation
        est_hm_list, encoding = self.net_hm(images)
        return est_hm_list, encoding

# FreiHand Decoder
class MyHandDecoder(nn.Module):
    def __init__(self,inp_neurons=1536,use_mean_shape=False):
        super(MyHandDecoder, self).__init__()
        self.hand_decode = MyPoseHand(inp_neurons=inp_neurons,use_mean_shape = use_mean_shape)
        if use_mean_shape:
            print("use mean MANO shape")
        else:
            print("do not use mean MANO shape")
        #self.hand_faces = self.hand_decode.mano_branch.faces

    def forward(self, features):
        #sides = torch.zeros(features.shape[0],1)
        #verts, faces, joints = self.hand_decode(features, Ks)
        '''
        joints, verts, faces, theta, beta = self.hand_decode(features)
        return joints, verts, faces, theta, beta
        '''
        joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses = self.hand_decode(features)
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses

class light_estimator(nn.Module):
    def __init__(self, dim_in=1536):
        super(light_estimator, self).__init__()
        self.fc1 = nn.Linear(dim_in, 256)
        self.fc2 = nn.Linear(256, 11)
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        lights = torch.sigmoid(self.fc2(x))
        return lights

class texture_light_estimator(nn.Module):
    def __init__(self, num_channel=32, dim_in=56,mode='surf'):
        super(texture_light_estimator, self).__init__()
        self.base_layers = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=10, stride=4, padding=1),#[48,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#[48,6,6]
            nn.Conv2d(48, 64, kernel_size=3),#[64,4,4]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#[64,2,2]
        )
        self.texture_reg = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1538*3),
        )
        self.light_reg = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11),
            #nn.Sigmoid()
        )
        self.mode = mode
        #self.texture_mean = torch.tensor([0.5, 0.5, 0.5])
        self.texture_mean = torch.tensor([200/256, 150/256, 150/256]).float()
        normal_init(self.texture_reg[0],std=0.001)
        normal_init(self.texture_reg[2],std=0.001)
        normal_init(self.light_reg[0],std=0.001)
        normal_init(self.light_reg[2],mean=1,std=0.001)

    def forward(self, low_features):
        base_features = self.base_layers(low_features)#[b,64,2,2]
        base_features = base_features.view(base_features.shape[0],-1)##[B,256]
        # texture
        bias = self.texture_reg(base_features)
        mean_t = self.texture_mean.to(device=bias.device)
        if self.mode == 'surf':
            bias = bias.view(-1, 1538, 3)#[b, 778, 3]
            mean_t = mean_t.unsqueeze(0).unsqueeze(0).repeat(1,bias.shape[1],1)#[1, 778, 3]
        #import pdb; pdb.set_trace()
        textures = mean_t + bias#[b,778,3]
        #import pdb; pdb.set_trace()
        # lighting
        lights = self.light_reg(base_features)#[b,11]
        #import pdb; pdb.set_trace()
        #textures = torch.clamp(textures,0,1)
        #textures = torch.clamp(textures,min=0)
        #lights = torch.clamp(lights,min=0)
        #import pdb; pdb.set_trace()
        return textures, lights

class heatmap_attention(nn.Module):
    def __init__(self, num_channel=256, dim_in=64, out_len=1536, mode='surf'):
        super(heatmap_attention, self).__init__()
        self.base_layers = nn.Sequential(
            nn.BatchNorm2d(num_channel),
            nn.Conv2d(num_channel, 64, kernel_size=10, stride=7, padding=1),#[64,9,9]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=3, padding=1),#[64,3,3]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),#[64,1,1]
        )
        self.reg = nn.Sequential(
            nn.Linear(64, out_len),
        )
        #import pdb; pdb.set_trace()
    def forward(self, x):
        x0 = self.base_layers(x)
        x0 = x0.view(x.shape[0],-1)#[b,64]
        return self.reg(x0)


class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        # 2D hand estimation
        if 'heatmaps' in args.train_requires or 'heatmaps' in args.test_requires:
            self.rgb2hm = RGB2HM()

        # 3D hand estimation
        if "joints" in args.train_requires or "verts" in args.train_requires or "joints" in args.test_requires or "verts" in args.test_requires:
            self.regress_mode = args.regress_mode#options: 'mano' 'hm2mano'
            self.use_mean_shape = args.use_mean_shape
            self.use_2d_as_attention = args.use_2d_as_attention
            if self.regress_mode == 'mano':# efficient-b3
                self.encoder = Encoder()
                self.dim_in = 1536
                self.hand_decoder = MyHandDecoder(inp_neurons=self.dim_in, use_mean_shape = self.use_mean_shape)
                if self.use_2d_as_attention:
                    self.heatmap_attention = heatmap_attention(out_len=self.dim_in)

            self.render_choice = args.renderer_mode
            self.texture_choice = args.texture_mode

            # Renderer & Texture Estimation & Light Estimation
            if self.render_choice == 'NR':
                # Define a neural renderer
                import neural_renderer as nr
                if 'lights' in args.train_requires or 'lights' in args.test_requires:
                    renderer_NR = nr.Renderer(image_size=args.image_size,background_color=[1,1,1],camera_mode='projection',orig_size=224,light_intensity_ambient=None, light_intensity_directional=None,light_color_ambient=None, light_color_directional=None,light_direction=None)#light_intensity_ambient=0.9
                else:
                    renderer_NR = nr.Renderer(image_size=args.image_size,camera_mode='projection',orig_size=224)
                #import pdb;pdb.set_trace()
                self.renderer_NR = renderer_NR

                '''
                if self.texture_choice == 'surf':
                    self.texture_estimator = TextureEstimator(dim_in=self.dim_in,mode='surfaces')
                elif self.texture_choice == 'nn_same':
                    self.color_estimator = ColorEstimator(dim_in=self.dim_in)
                self.light_estimator = light_estimator(dim_in=self.dim_in)
                '''

                self.texture_light_from_low = texture_light_estimator(mode='surf')
                #[print(aa.requires_grad) for aa in self.encoder.parameters()]
            # Pose adapter
            self.use_pose_regressor = args.use_pose_regressor
            if (args.train_datasets)[0] == 'FreiHand':
                self.get_gt_depth = True
                self.dataset = 'FreiHand'
            elif (args.train_datasets)[0] == 'RHD':
                self.get_gt_depth = False
                self.dataset = 'RHD'
                if self.use_pose_regressor:
                    self.mesh2pose = mesh2poseNet()
            elif (args.train_datasets)[0] == 'Obman':
                self.get_gt_depth = False
                self.dataset = 'Obman'
            elif (args.train_datasets)[0] == 'HO3D':
                self.get_gt_depth = True
                self.dataset = 'HO3D'
                #Check
            else:
                self.get_gt_depth = False
            

            # Perception loss
            '''
            if 'percep' in args.losses:
                self.hand_det = handpose_model()
                model_dict = util.transfer(self.hand_det, torch.load('/dockerdata/terrchen/code/pytorch-openpose/model/hand_pose_model.pth'))
                print("Read openpose weights.")
                self.hand_det.load_state_dict(model_dict)
                self.hand_det.eval()
            '''
            if 'percep_feat' in args.train_requires:#'percep' in args.losses:
                '''
                self.percep_encoder = Percep_Encoder()
                for param in self.percep_encoder.parameters():
                    param.requires_grad = False
                '''
                '''
                from losses import VGGPerceptualLoss,EffiPerceptualLoss
                self.perc_crit = EffiPerceptualLoss(pretrained_model=args.efficientnet_pretrained)#VGGPerceptualLoss()
                '''
                from losses import OpenPosePerceptualLoss
                self.perc_crit = OpenPosePerceptualLoss(model_path='/apdcephfs/private_terrchen/mycode/pytorch-openpose/model/hand_pose_model.pth')

                # [a for a in self.perc_crit.percep_encoder.parameters()][0][0]
        else:
            self.regress_mode = None
        #import pdb; pdb.set_trace()
        #import numpy as np
        #np.sum([p.numel() for p in model.parameters()]).item()
    def predict_singleview(self, images, mask_images, Ks, task, requires, gt_verts, bgimgs):
        vertices, faces, joints, shape, pose, trans, segm_out, textures, lights = None, None, None, None, None, None, None, None, None
        re_images, re_sil, re_img, re_depth, gt_depth = None, None, None, None, None
        pca_text, face_textures = None, None
        output = {}
        # 1. Heat-map estimation
        #end = time.time()
        if self.regress_mode == 'hm2mano' or task == 'hm_train' or 'heatmaps' in requires:
            images_this = images
            if images_this.shape[3] != 256:
                pad = nn.ZeroPad2d(padding=(0,32,0,32))
                #import pdb; pdb.set_trace()
                images_this = pad(images_this)#[b,3,256,256]
            est_hm_list, encoding = self.rgb2hm(images_this)
            
            # est_hm_list: len() 2; [b, 21, 64, 64]
            # this is not well differentiable
            #est_pose_uv = util.compute_uv_from_heatmaps(est_hm_list[-1], images_this.shape[2:4])#images.shape[2:4] torch.Size([224, 224]))  # B x K x 3
            est_pose_uv_list = []
            for est_hm in est_hm_list:
                est_pose_uv = util.compute_uv_from_integral(est_hm, images_this.shape[2:4])#check
                est_pose_uv_list.append(est_pose_uv)
            
            output['hm_list'] = est_hm_list
            #output['hm_pose_uv'] = est_pose_uv#[b,21,3]
            output['hm_pose_uv_list'] = est_pose_uv_list
            output['hm_j2d_list'] = [hm_pose_uv[:,:,:2] for hm_pose_uv in est_pose_uv_list]
        if task == 'hm_train': 
            #return est_pose_uv, est_hm_list
            return output
        else:
            if self.regress_mode == 'hm2mano':
                # 2. Hand shape and pose estimate
                joints, vertices, faces, pose, shape, features = self.hm2hand(est_hm_list, encoding)
                # joints: [b,21,3]; vertices: [b,778,3]; faces: [b,1538,3]; 
                # pose: [b,6]; shape: [b,10]; features: [b,4096]; 
            elif self.regress_mode == 'mano' or self.regress_mode == 'mano1':
                features, low_features = self.encoder(images)#[b,1536]
                
                if self.use_2d_as_attention:
                    attention_2d = self.heatmap_attention(encoding[-1])
                    features = torch.mul(features, attention_2d)
                #import pdb; pdb.set_trace()

                #import pdb;pdb.set_trace()
                if 'joints' in requires or 'verts' in requires:
                    #joints, vertices, faces, pose, shape = self.hand_decoder(features)
                    joints, vertices, faces, pose, shape, scale, trans, rot, tsa_poses  = self.hand_decoder(features)
                    if self.dataset == 'RHD' and self.use_pose_regressor:
                        joints_res = self.mesh2pose(vertices)
                        #import pdb; pdb.set_trace()
                        joints = joints + joints_res
            #print(time.time() - end)
            #print('Time {batch_time.val:.0f}\t'.format(batch_time))   
            output['joints'] = joints
            output['vertices'] = vertices
            output['pose'] = pose
            output['shape'] = shape
            output['scale'] = scale
            output['trans'] = trans
            output['rot'] = rot
            output['tsa_poses'] = tsa_poses
            
            #import pdb; pdb.set_trace()
            # 3. Texture & Lighting Estimation
            if 'textures' in requires or 'lights' in requires:
                #low_features.requires_grad = True
                #end = time.time()
                textures, lights = self.texture_light_from_low(low_features)
                #print(time.time() - end)
                #import pdb; pdb.set_trace()
                if 'lights' in requires:                     
                    self.renderer_NR.light_intensity_ambient = lights[:,0].to(vertices.device)
                    self.renderer_NR.light_intensity_directional = lights[:,1].to(vertices.device)
                    self.renderer_NR.light_color_ambient = lights[:,2:5].to(vertices.device)
                    self.renderer_NR.light_color_directional = lights[:,5:8].to(vertices.device)
                    self.renderer_NR.light_direction = lights[:,8:].to(vertices.device)
                    '''
                    self.renderer_NR.light_intensity_ambient = lights[:,0].to(vertices.device)
                    self.renderer_NR.light_color_ambient = torch.ones_like(lights[:,2:5]).to(vertices.device)
                    self.renderer_NR.light_intensity_directional = lights[:,1].to(vertices.device)
                    self.renderer_NR.light_color_directional = torch.ones_like(lights[:,5:8]).to(vertices.device)
                    self.renderer_NR.light_direction = lights[:,8:].to(vertices.device)
                    '''
                output['textures'] = textures
                output['lights'] = lights
                #import pdb;pdb.set_trace()

            #del features
            #import pdb; pdb.set_trace()
            # 4. Render image
            faces = faces.type(torch.int32)
            if self.render_choice == 'NR':
                # use neural renderer
                #I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
                #Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).to(Ks.device)
                # create textures
                if textures is None:
                    texture_size = 1
                    textures = torch.ones(faces.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(vertices.device)
                
                self.renderer_NR.R = torch.unsqueeze(torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float(),0).repeat(Ks.shape[0],1,1).to(vertices.device)
                self.renderer_NR.t = torch.unsqueeze(torch.tensor([[0,0,0]]).float(),0).repeat(Ks.shape[0],1,1).to(vertices.device)
                self.renderer_NR.K = Ks[:,:,:3].to(vertices.device)
                self.renderer_NR.dist_coeffs = self.renderer_NR.dist_coeffs.to(vertices.device)
                #import pdb; pdb.set_trace()
                
                face_textures = textures.view(textures.shape[0],textures.shape[1],1,1,1,3)
                
                re_img,re_depth,re_sil = self.renderer_NR(vertices, faces, torch.tanh(face_textures), mode=None)

                re_depth = re_depth * (re_depth < 1).float()#set 100 into 0

                #import pdb; pdb.set_trace()
                if self.get_gt_depth and gt_verts is not None:
                    gt_depth = self.renderer_NR(gt_verts, faces, mode='depth')
                    gt_depth = gt_depth * (gt_depth < 1).float()#set 100 into 0
                #import pdb; pdb.set_trace()
            
            output['faces'] = faces
            output['re_sil'] = re_sil
            output['re_img'] = re_img
            output['re_depth'] = re_depth
            output['gt_depth'] = gt_depth
            if re_sil is not None:
                output['maskRGBs'] = images.mul((re_sil>0).float().unsqueeze(1).repeat(1,3,1,1))
            output['face_textures'] = face_textures
            output['render'] = self.renderer_NR
            #output[''] = 
            # Perceptual calculation
            if 'percep_feat' in requires and re_img is not None:
                # only use foreground part
                #import pdb; pdb.set_trace()
                
                #perc_loss = self.perc_crit(torch.mul(images,re_sil.detach().unsqueeze(1)),re_img)
                #perc_features = self.perc_crit.extract_features(torch.mul(images,re_sil.detach().unsqueeze(1)),re_img)
                #in_percep, in_percep_low = self.percep_encoder(torch.mul(images,re_sil.detach().unsqueeze(1)))
                #out_percep, out_percep_low = self.percep_encoder(re_img)
                #perc_features = self.perc_crit(images,re_img)
                perc_loss = self.perc_crit(images,re_img)
                # [a for a in self.perc_crit.model.model1_0.parameters()][0][0]
                #import pdb; pdb.set_trace()
                '''
                iii = 0
                for name,parameters in self.percep_encoder.named_parameters():
                    if iii == 0:
                        print(name,':',parameters.size())
                        print(parameters[0])
                    iii += 1
                #import pdb; pdb.set_trace()
                '''
                #output['in_percep'] = in_percep
                #output['out_percep'] = out_percep
                output['perc_loss'] = perc_loss
                #output['perc_features'] = perc_features
            # Network stacked
            if 'stacked' in requires:
                
                new_img = torch.where(re_img>0,re_img,bgimgs).detach()
                #import pdb;pdb.set_trace()
                
                if self.regress_mode == 'mano':
                    features_s, low_features_s = self.encoder(new_img)#[b,1536]
                    textures_s, lights_s = self.texture_light_from_low(low_features_s)
                    crit = nn.L1Loss()
                    light_s_loss = crit(lights_s, lights.detach())
                    textures_s_loss = crit(textures_s, textures.detach())
                    output['light_s_loss'] = light_s_loss
                    output['textures_s_loss'] = textures_s_loss
                    joints_s, vertices_s, faces_s, pose_s, shape_s, scale_s, trans_s, rot_s, tsa_poses_s  = self.hand_decoder(features_s)
                    crit_mse = nn.MSELoss()
                    joints_s_loss = crit_mse(joints_s, joints.detach())
                    vertices_s_loss = crit_mse(vertices_s, vertices.detach())
                    output['joints_s_loss'] = joints_s_loss
                    output['vertices_s_loss'] = vertices_s_loss
                    output['tsa_poses_s'] = tsa_poses_s
                    faces_s = faces_s.type(torch.int32)
                    face_textures_s = textures_s.view(textures_s.shape[0],textures_s.shape[1],1,1,1,3)
                    re_img_s,re_depth_s,re_sil_s = self.renderer_NR(vertices_s, faces_s, torch.tanh(face_textures_s), mode=None)
                    re_depth_s = re_depth_s * (re_depth_s < 1).float()#set 100 into 0
                    photo_s_loss = crit(re_img_s,re_img.detach())
                    depth_s_loss = crit(re_depth_s,re_depth.detach())
                    sil_s_loss = crit(re_sil_s,re_sil.detach())
                    # ss photometric
                    mask_img_s = images.mul(re_sil_s.unsqueeze(1)).detach()
                    photo_ss_loss = crit(re_img_s,mask_img_s)
                    output['photo_s_loss'] = photo_s_loss
                    output['depth_s_loss'] = depth_s_loss
                    output['sil_s_loss'] = sil_s_loss
                    output['photo_ss_loss'] = photo_ss_loss

                    # feature consistency
                    feature_perceptual_loss = crit(features, features_s)#check detach??
                    output['feature_perceptual_loss'] = feature_perceptual_loss
                    #import pdb; pdb.set_trace()
                    output['vertices_s'] = vertices_s
                    output['joints_s'] = joints_s
                    output['faces_s'] = faces_s
                    output['re_img_s'] = re_img_s
                    output['re_depth_s'] = re_depth_s
                    output['re_sil_s'] = re_sil_s
                    output['new_img'] = new_img
                    output['new_maskimg'] = mask_img_s
            
            '''
            if 'feats' in requires and re_img is not None:
                feats_mask = self.encoder(mask_images)
            #import pdb; pdb.set_trace()
            if 'percep_feat' in requires and re_img is not None:
                #import pdb; pdb.set_trace()
                #images = images.permute(0,2,3,1)*255
                cv_images = torch.zeros_like(images)
                cv_images[:,0,:,:]=images[:,2,:,:]
                cv_images[:,1,:,:]=images[:,1,:,:]
                cv_images[:,2,:,:]=images[:,0,:,:]
                
                #images = re_img.permute(0,2,3,1)*255
                #import pdb; pdb.set_trace()
                re_img_mix = torch.where(re_img>0,re_img,images)
                #re_img_mix = re_img
                cv_re_img = torch.zeros_like(re_img)
                cv_re_img[:,0,:,:]=re_img_mix[:,2,:,:]
                cv_re_img[:,1,:,:]=re_img_mix[:,1,:,:]
                cv_re_img[:,2,:,:]=re_img_mix[:,0,:,:]
                
                in_percep = self.hand_det(cv_images)# cv_images: GBR 0-255 [b,3,224,224]
                #
                out_percep = self.hand_det(cv_re_img)# [b,22,28,28]
                # mse of in-percep and out_percep is e-17!
                #import pdb; pdb.set_trace()
                del cv_images, cv_re_img, images
            '''

            '''
            if task == 'stacked_train':
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, features
            elif 'feats' in requires:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, features, feats_mask
            elif 'percep_feat' in requires:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, textures, in_percep, out_percep
            elif 'scalerot' in requires:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, textures, pca_text, scale, rot, tsa_poses
            else:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, textures, pca_text, tsa_poses#, scale, rot
            '''
            return output
    def forward(self, images=None, mask_images = None, viewpoints=None, P=None, voxels=None, mano_para = None, task='train', requires=['joints'], gt_verts=None, gt_2d_joints=None, bgimgs=None):
        if task == 'train' or task == 'hm_train':
            return self.predict_singleview(images, mask_images, P, task, requires, gt_verts, bgimgs)
        elif task == 'stacked_train':
            return self.stacked_predict_singleview(images, mask_images, P, task, requires, gt_verts, bgimgs)
        elif task == 'test':
            return self.evaluate_iou(images, voxels)