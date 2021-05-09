import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from utils.fh_utils import proj_func
from losses import bone_direction_loss, tsa_pose_loss#image_l1_loss, iou_loss, ChamferLoss,
from utils.laplacianloss import LaplacianLoss

import os
import matplotlib.pyplot as plt
import numpy as np

from draw_util import draw_2d_error_curve

import torch.optim as optim
from utils.hand_3d_model import rot_pose_beta_to_mesh
from utils.fh_utils import Mano2Frei, RHD2Frei, HO3D2Frei, Frei2HO3D, AverageMeter
from losses import tsa_pose_loss, bone_direction_loss
import time
import util
import json
import pytorch_ssim

def data_dic(data_batch, dat_name, set_name, args) -> dict:
    example_torch = {}
    if dat_name == 'FreiHand':
        # raw data
        #import pdb; pdb.set_trace()
        if "trans_images" in data_batch:
            imgs = (data_batch['trans_images']).cuda()#[b,3,224,224]
        elif "images" in data_batch:
            imgs = (data_batch['images']).cuda()
        else:
            import pdb; pdb.set_trace()
        example_torch['imgs'] = imgs

        # refhand image
        if "refhand" in data_batch:
            example_torch['refhand'] = data_batch["refhand"]

        if "trans_Ks" in data_batch:
            Ks = data_batch['trans_Ks'].cuda()#[b,3,3]
        elif "Ks" in data_batch:
            Ks = data_batch['Ks'].cuda()#[b,3,3]
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'scales' in data_batch.keys():
            example_torch['scales'] = data_batch['scales'].float()#[b]
        
        #j2d_gt, masks, maskRGBs, manos, verts, joints, open_2dj, CRFmasks, CRFmaskRGBs = None, None, None, None, None, None, None, None, None
        example_torch['idxs'] = data_batch['idxs'].cuda()
        
        if 'CRFmasks' in data_batch.keys():
            CRFmasks = data_batch['CRFmasks'].cuda()
            example_torch['CRFmasks'] = CRFmasks
            example_torch['CRFmaskRGBs'] = imgs.mul(CRFmasks[:,2].unsqueeze(1).repeat(1,3,1,1).float())#
        if 'open_2dj' in data_batch.keys() or 'trans_open_2dj' in data_batch.keys():
            if 'trans_open_2dj' in data_batch.keys():
                example_torch['open_2dj'] = data_batch['trans_open_2dj'].cuda()
            elif 'open_2dj' in data_batch.keys():
                example_torch['open_2dj'] = data_batch['open_2dj'].cuda()# idx(openpose) == idx(freihand)
            open_2dj_con = data_batch['open_2dj_con'].cuda()
            # check!
            #import pdb; pdb.set_trace()
            texture_idx_con = ((data_batch['idxs']<32560).float()+0.1).cuda()
            texture_con = torch.mean((torch.min(open_2dj_con.squeeze(),1)[0]>0.1).float().unsqueeze(1).mul(open_2dj_con.squeeze()),1)#[b]
            texture_con = torch.mul(texture_con,texture_idx_con)
            example_torch['open_2dj_con'] = open_2dj_con
            example_torch['texture_con'] = texture_con
            #open_2dj_con = (torch.min(open_2dj_con.squeeze(),1)[0]>0).float().unsqueeze(1).mul(open_2dj_con.squeeze()).unsqueeze(-1)
            '''
            idxs_con = ((idxs<32560).float()+1)/2
            texture_con = idxs_con.mul(texture_con)
            '''
        if 'training' in set_name:
            if 'masks' in data_batch.keys():
                masks = data_batch['masks'].cuda()#[b,3,224,224]
                example_torch['masks'] = masks
                #maskRGBs = data_batch['maskRGBs'].cuda()#[b,3,224,224]
                segms_gt = masks[:,0].long()#[b, 224, 224]# mask_gt
                example_torch['segms_gt'] = segms_gt
            
            if 'trans_masks' in data_batch.keys():
                masks = data_batch['trans_masks'].cuda()#[b,3,224,224]
                example_torch['masks'] = masks
                #maskRGBs = data_batch['maskRGBs'].cuda()#[b,3,224,224]
                segms_gt = masks[:,0].long()#[b, 224, 224]# mask_gt
                example_torch['segms_gt'] = segms_gt
            
            if 'manos' in data_batch.keys():
                manos = torch.squeeze(data_batch['manos'],1).cuda()#[b,61]
                example_torch['manos'] = manos
            if 'joints' in data_batch.keys():
                joints = data_batch['joints'].cuda()#[b,21,3]
                example_torch['joints'] = joints
                j2d_gt = proj_func(joints, Ks)
                example_torch['j2d_gt'] = j2d_gt
            if 'trans_joints' in data_batch.keys():
                joints = data_batch['trans_joints'].cuda()#[b,21,3]
                example_torch['joints'] = joints
                j2d_gt = proj_func(joints, Ks)
                example_torch['j2d_gt'] = j2d_gt
            if 'verts' in data_batch.keys():
                verts = data_batch['verts'].cuda()
                example_torch['verts'] = verts
            if "trans_verts" in data_batch.keys():
                verts = data_batch['trans_verts'].cuda()
                example_torch['verts'] = verts
            #import pdb; pdb.set_trace()
            if args.semi_ratio is not None and 'j2d_gt' in example_torch and 'open_2dj' in example_torch:
                raw_idx = example_torch['idxs']%32560
                mix_open_2dj = torch.where((raw_idx<32560*args.semi_ratio).view(-1,1,1),example_torch['j2d_gt'],example_torch['open_2dj'])
                mix_open_2dj_con = torch.where((raw_idx<32560*args.semi_ratio).view(-1,1,1),torch.ones_like(example_torch['open_2dj_con']).to(example_torch['open_2dj_con'].device),example_torch['open_2dj_con'])
                example_torch['open_2dj'] = mix_open_2dj
                example_torch['open_2dj_con'] = mix_open_2dj_con
                
    elif dat_name == 'HO3D0':
        example_torch['imgs'] = (data_batch['img_crop']).cuda()#[b,3,224,224]
        Ks = data_batch['K_crop'].cuda()#[b,3,3]
        # only for HO3D
        Ks = Ks.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())# check

        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'uv21_crop' in data_batch.keys():
            j2d_gt = data_batch['uv21_crop'].cuda()
            j2d_gt = HO3D2Frei(j2d_gt)
            example_torch['j2d_gt'] = j2d_gt
        if 'xyz21' in data_batch.keys():
            joints = data_batch['xyz21'].cuda()#[b,21,3]
            joints = HO3D2Frei(joints)
            # only for HO3D
            joints = joints.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            example_torch['joints'] = joints
            j2d_gt_proj = proj_func(joints, Ks)
            '''
            
            Ks_raw = data_batch['camMat'].cuda()
            Ks_raw = Ks_raw.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            j2d_gt_raw = proj_func(joints, Ks_raw)
            import pdb; pdb.set_trace()
            crop_center = data_batch['crop_center0']
            image_center = torch.tensor([320,240]).float().unsqueeze(0).repeat(crop_center.shape[0],1)
            trans_image = image_center - crop_center
            scale = (Ks_raw[:,0,0]+Ks_raw[:,1,1])/2 torch.mean(joints,1)[:,2]
            trans_xy = trans_image.cuda()*torch.mean(joints,1)[:,2].unsqueeze(-1)*torch.reciprocal((Ks_raw[:,0,0]+Ks_raw[:,1,1])/2).unsqueeze(-1)
            joints_trans = joints + torch.cat((trans_xy, torch.zeros([trans_xy.shape[0],1]).to(trans_xy.device)),1).unsqueeze(1)
            proj_func(joints_trans, Ks_raw)
            #example_torch['j2d_gt'] = j2d_gt
            '''
        if 'open_2dj_crop' in data_batch.keys():
            example_torch['open_2dj'] = data_batch['open_2dj_crop'].cuda()# idx(openpose) == idx(freihand)
            open_2dj_con = data_batch['open_2dj_con'].cuda()
            # check!q
            texture_con = torch.mean(open_2dj_con.squeeze(),1)
            example_torch['open_2dj_con'] = open_2dj_con
            example_torch['texture_con'] = texture_con
    elif dat_name == 'HO3D':
        example_torch['imgs'] = torch_f.interpolate(data_batch['img_crop'],(224,224)).cuda()#[b,3,640,640] --> [b,3,224,224]
        Ks = data_batch['K_crop'].cuda()#[b,3,3]
        # only for HO3D
        Ks = Ks.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())# check
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'uv21_crop' in data_batch.keys():
            j2d_gt = data_batch['uv21_crop'].float().cuda()
            j2d_gt = HO3D2Frei(j2d_gt)
            example_torch['j2d_gt'] = j2d_gt
        if 'xyz21' in data_batch.keys():
            joints = data_batch['xyz21'].cuda()#[b,21,3]
            joints = HO3D2Frei(joints)
            # only for HO3D
            joints = joints.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            example_torch['joints'] = joints
            #j2d_gt_proj = proj_func(joints, Ks)
            #example_torch['j2d_gt'] = j2d_gt_proj
            
            #joints0 = data_batch['xyz210'].cuda()#[b,21,3]
            #joints0 = HO3D2Frei(joints0)
            # only for HO3D
            #joints0 = joints0.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            #j2d_gt_proj0 = proj_func(joints0, Ks)
            #example_torch['joints'] = joints0
            #example_torch['j2d_gt'] = j2d_gt_proj0
        if 'open_2dj_crop' in data_batch.keys():
            example_torch['open_2dj'] = data_batch['open_2dj_crop'].float().cuda()# idx(openpose) == idx(freihand)
            open_2dj_con = data_batch['open_2dj_con'].cuda()
            # check!q
            texture_con = torch.mean(open_2dj_con.squeeze(),1)
            example_torch['open_2dj_con'] = open_2dj_con
            example_torch['texture_con'] = texture_con
        #import pdb; pdb.set_trace()

    elif dat_name == 'RHD':
        '''
        imgs = data_batch['images'].cuda()#[b,3,320,320]
        Ks = data_batch['Ks'].cuda()
        uv21 = data_batch['uv21'].cuda()
        '''
        # for croped
        imgs = data_batch['img_crop'].cuda()#[b,3,224,224]
        example_torch['imgs'] = imgs
        Ks = data_batch['K_crop'].cuda()#[b,3,3]
        I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
        Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).cuda()
        example_torch['Ps'] = torch.bmm(Ks, Is)#.cuda()
        example_torch['Ks'] = Ks
        if 'uv21_crop' in data_batch:
            j2d_gt = data_batch['uv21_crop'].cuda()#[B, 21, 2]
            j2d_gt = RHD2Frei(j2d_gt)
            example_torch['j2d_gt']=j2d_gt
        if 'xyz21' in data_batch:
            joints = data_batch['xyz21'].cuda()#[B, 21, 3]
            joints = RHD2Frei(joints)
            example_torch['joints']=joints
        '''
        if 'xyz_trans' in data_batch and 'K_crop_trans' in data_batch:
            joints_trans = data_batch['xyz_trans'].cuda()#[B, 21, 3]
            joints_trans = RHD2Frei(joints_trans)
            example_torch['joints_trans']=joints_trans
            j2d_gt_trans = proj_func(joints_trans, data_batch['K_crop_trans'].cuda())
            import pdb; pdb.set_trace()
        '''
        # Calculate projected 2D joints
        #j2d_syngt = proj_func(joints, Ks)
        if 'keypoint_scale' in data_batch:
            keypoint_scale = data_batch['keypoint_scale'].cuda()#[B]
            example_torch['keypoint_scale'] = keypoint_scale
        if "uv_vis" in data_batch:
            uv_vis = data_batch['uv_vis']#[B,21] True False
            uv_vis = RHD2Frei(uv_vis)
            example_torch['uv_vis'] = uv_vis
        '''
        if 'mask_crop' in data_batch:
            masks = data_batch['mask_crop'].cuda()#[B,1,224,224] 0 1
            masks = masks.repeat(1,3,1,1).float()#[B,3,224,224] 0 1
            maskRGBs = imgs.mul(masks)#
        '''
        if 'sides' in data_batch:
            # sides before flip to right
            side = data_batch['sides'].cuda()#[8,1]  0 left; 1 right
        
        if 'open_2dj' in data_batch.keys():
            example_torch['open_2dj'] = data_batch['open_2dj_crop'].cuda()# idx(openpose) == idx(freihand)
            example_torch['open_2dj_con'] = data_batch['open_2dj_con'].cuda()
    #import pdb; pdb.set_trace()
    return example_torch
        
def loss_func(examples, outputs, dat_name, args):
    loss_dic = {}
    # heatmap loss
    if 'hm_integral' in args.losses and ('open_2dj' in examples) and ('open_2dj_con' in examples) and ('hm_j2d_list' in outputs):
        hm_j2d_list = outputs['hm_j2d_list']
        hm_integral_loss = torch.zeros(1).to(hm_j2d_list[-1].device)
        for hm_j2d in hm_j2d_list:
            hm_2dj_distance = torch.sqrt(torch.sum((examples['open_2dj']-hm_j2d)**2,2))#[b,21]
            open_2dj_con_hm = examples['open_2dj_con'].squeeze(2)
            hm_integral_loss += (torch.sum(hm_2dj_distance.mul(open_2dj_con_hm**2))/torch.sum((open_2dj_con_hm**2)))#check
        loss_dic['hm_integral'] = args.lambda_hm * hm_integral_loss
    else:
        loss_dic['hm_integral'] = torch.zeros(1)
    #import pdb; pdb.set_trace()
    
    if 'hm_integral_gt' in args.losses and ('j2d_gt' in examples) and ('hm_j2d_list' in outputs):
        hm_j2d_list = outputs['hm_j2d_list']
        hm_integral_loss = torch.zeros(1).to(hm_j2d_list[-1].device)
        for hm_j2d in hm_j2d_list:
            hm_2dj_distance0 = torch.sqrt(torch.sum((examples['j2d_gt']-hm_j2d)**2,2))#[b,21]
            open_2dj_con_hm0 = torch.ones_like(hm_2dj_distance0)
            hm_integral_loss += torch.sum(hm_2dj_distance0.mul(open_2dj_con_hm0**2))/torch.sum((open_2dj_con_hm0**2))
        loss_dic['hm_integral_gt'] = args.lambda_hm * hm_integral_loss
    else:
        loss_dic['hm_integral_gt'] = torch.zeros(1)
    
    # check
    '''
    if args.frei_selfsup and dat_name=="FreiHand":
        loss_dic['hm_integral_gt'] = torch.zeros(1)
    '''
    #import pdb; pdb.set_trace()
    # 2d joint loss
    if 'j2d_gt' in examples and ('j2d' in outputs):
        joint_2d_loss = torch_f.mse_loss(examples['j2d_gt'], outputs['j2d'])
        joint_2d_loss = args.lambda_j2d_gt * joint_2d_loss
        loss_dic['joint_2d'] = joint_2d_loss
    else:
        loss_dic['joint_2d'] = torch.zeros(1)

    # open pose 2d joint loss
    if 'open_2dj' in args.losses and ('open_2dj' in examples) and ('open_2dj_con' in examples) and ('j2d' in outputs):
        open_2dj_distance = torch.sqrt(torch.sum((examples['open_2dj']-outputs['j2d'])**2,2))
        open_2dj_distance = torch.where(open_2dj_distance<5, open_2dj_distance**2/10,open_2dj_distance-2.5)
        keypoint_weights = torch.tensor([[2,1,1,1,1.5,1,1,1,1.5,1,1,1,1.5,1,1,1,1.5,1,1,1,1.5]]).to(open_2dj_distance.device).float()
        open_2dj_con0 = examples['open_2dj_con'].squeeze(2)
        open_2dj_con0 = open_2dj_con0.mul(keypoint_weights)
        open_2dj_loss = (torch.sum(open_2dj_distance.mul(open_2dj_con0**2))/torch.sum((open_2dj_con0**2)))
        open_2dj_loss = args.lambda_j2d * open_2dj_loss
        loss_dic['open_2dj'] = open_2dj_loss
    else:
        loss_dic['open_2dj'] = torch.zeros(1)
    #import pdb; pdb.set_trace()
    # open pose 2d joint loss --- Downgrade Version
    if "open_2dj_de" in args.losses and ('open_2dj' in examples) and ('j2d' in outputs):
        open_2dj_loss = torch_f.mse_loss(examples['open_2dj'],outputs['j2d'])
        open_2dj_loss = args.lambda_j2d_de * open_2dj_loss
        loss_dic["open_2dj_de"] = open_2dj_loss
    else:
        loss_dic["open_2dj_de"] = torch.zeros(1)

    # 3D joint loss & Bone scale loss
    
    if 'joints' in outputs and 'joints' in examples:
        joint_3d_loss = torch_f.mse_loss(outputs['joints'], examples['joints'])
        joint_3d_loss = args.lambda_j3d * joint_3d_loss
        loss_dic["joint_3d"] = joint_3d_loss
        joint_3d_loss_norm = torch_f.mse_loss((outputs['joints']-outputs['joints'][:,9].unsqueeze(1)),(examples['joints']-examples['joints'][:,9].unsqueeze(1)))
        joint_3d_loss_norm = args.lambda_j3d_norm * joint_3d_loss_norm
        loss_dic["joint_3d_norm"] = joint_3d_loss_norm
    else:
        loss_dic["joint_3d"] = torch.zeros(1)
        loss_dic["joint_3d_norm"] = torch.zeros(1)

    # bone direction loss
    if 'open_bone_direc' in args.losses and ('open_2dj' in examples) and ('open_2dj_con' in examples) and ('j2d' in outputs):
        open_bone_direc_loss = bone_direction_loss(outputs['j2d'], examples['open_2dj'], examples['open_2dj_con'])
        open_bone_direc_loss = args.lambda_bone_direc * open_bone_direc_loss
        loss_dic['open_bone_direc'] = open_bone_direc_loss
    else:
        loss_dic['open_bone_direc'] = torch.zeros(1)
    
    if 'bone_direc' in args.losses and ('j2d_gt' in examples) and ('j2d' in outputs):
        j2d_con = torch.ones_like(examples['j2d_gt'][:,:,0]).unsqueeze(-1)
        bone_direc_loss = bone_direction_loss(outputs['j2d'], examples['j2d_gt'], j2d_con)
        bone_direc_loss = args.lambda_bone_direc * bone_direc_loss
        loss_dic['bone_direc'] = bone_direc_loss
    else:
        loss_dic['bone_direc'] = torch.zeros(1)
    
    # 2d-3d keypoints consistency loss
    if ('hm_j2d_list' in outputs) and ('j2d' in outputs):
        hm_j2d_list = outputs['hm_j2d_list']
        kp_cons_distance = torch.sqrt(torch.sum((hm_j2d_list[-1]-outputs['j2d'])**2,2))
        kp_cons_distance = torch.where(kp_cons_distance<5, kp_cons_distance**2/10,kp_cons_distance-2.5)
        kp_cons_loss = torch.mean(kp_cons_distance)
        kp_cons_loss = args.lambda_kp_cons * kp_cons_loss
        loss_dic['kp_cons'] = kp_cons_loss
    else:
        loss_dic['kp_cons'] = torch.zeros(1)

    # mean scale regularization term
    if 'mscale' in args.losses and ('joints' in outputs):# and "joints" not in examples:
        out_bone_length = torch.sqrt(torch.sum((outputs['joints'][:,9, :] - outputs['joints'][:,10, :])**2,1))#check
        #import pdb;pdb.set_trace()
        crit = nn.L1Loss()
        mscale_loss = crit(out_bone_length,torch.ones_like(out_bone_length)*0.0282)#check
        mscale_loss = args.lambda_mscale * mscale_loss
        loss_dic['mscale'] = mscale_loss
    else:
        loss_dic['mscale'] = torch.zeros(1)
    
    # GT scale loss
    if 'scale' in args.losses and ('joints' in outputs) and 'scales' in examples:
        if dat_name == 'FreiHand':
            cal_scale = torch.sqrt(torch.sum((outputs['joints'][:,9]-outputs['joints'][:,10])**2,1))
            scale_loss = torch_f.mse_loss(cal_scale, examples['scales'].to(cal_scale.device))
            scale_loss = args.lambda_scale * scale_loss
            loss_dic['scale'] = scale_loss
    else:
        loss_dic['scale'] = torch.zeros(1)
    #import pdb; pdb.set_trace()
    # MANO pose regularization terms
    if 'tsa_poses' in outputs:
        pose_loss = tsa_pose_loss(outputs['tsa_poses'])
        pose_loss = args.lambda_pose * pose_loss
        loss_dic['tsa_poses'] = pose_loss
    else:
        loss_dic['tsa_poses'] = torch.zeros(1)
    
    # mesh texture regularization terms
    if 'mtex' in args.losses and ('textures' in outputs) and ('texture_con' in examples):
        textures = outputs['textures']
        std = torch.std(textures.view(textures.shape[0],-1,3),dim=1)#[b,3]
        mean = torch.mean(textures.view(textures.shape[0],-1,3),dim=1)
        textures_reg = (torch.where(textures>(mean.view(-1,1,1,1,1,3)+2*std.view(-1,1,1,1,1,3)),textures-mean.view(-1,1,1,1,1,3),torch.zeros_like(textures))+torch.where(textures<(mean.view(-1,1,1,1,1,3)-2*std.view(-1,1,1,1,1,3)),-textures+mean.view(-1,1,1,1,1,3),torch.zeros_like(textures))).squeeze()
        textures_reg = torch.sum(torch.mean(torch.mean(torch.mean(textures_reg,1),1),1).mul(examples['texture_con']*2))/torch.sum(examples['texture_con']**2)
        textures_reg = args.lambda_tex_reg * textures_reg
        loss_dic['mtex'] = textures_reg
    else:
        loss_dic['mtex'] = torch.zeros(1)

    # photometric loss
    if 're_img' in outputs and ('re_sil' in outputs) and ('texture_con' in examples):
        maskRGBs = outputs['maskRGBs']#examples['imgs'].mul((outputs['re_sil']>0).float().unsqueeze(1).repeat(1,3,1,1))
        re_img = outputs['re_img']
        crit = nn.L1Loss()
        #texture_loss = crit(re_img, maskRGBs).cpu()
        texture_con_this = examples['texture_con'].view(-1,1,1,1).repeat(1,re_img.shape[1],re_img.shape[2],re_img.shape[3])
        texture_loss = (torch.sum(torch.abs(re_img-maskRGBs).mul(texture_con_this**2))/torch.sum((texture_con_this**2)))
        texture_loss = args.lambda_texture * texture_loss
        loss_dic['texture'] = texture_loss
        #loss_mean_rgb = torch_f.mse_loss(torch.mean(maskRGBs),torch.mean(re_img)).cpu()
        loss_mean_rgb = (torch.sum(torch.abs(torch.mean(re_img.view(re_img.shape[0],-1),1)-torch.mean(maskRGBs.view(maskRGBs.shape[0],-1),1)).mul(examples['texture_con']**2))/torch.sum((examples['texture_con']**2)))
        loss_mean_rgb = args.lambda_mrgb * loss_mean_rgb
        loss_dic['mrgb'] = loss_mean_rgb
        ssim_tex = pytorch_ssim.ssim(re_img, maskRGBs)
        loss_ssim_tex = 1 - ssim_tex
        loss_ssim_tex = args.lambda_ssim_tex * loss_ssim_tex
        loss_dic['ssim_tex'] = loss_ssim_tex
        ssim_tex_depth = pytorch_ssim.ssim(re_img, outputs['re_depth'].unsqueeze(1).repeat(1,3,1,1))
        loss_ssim_tex_depth = 1 - ssim_tex_depth
        loss_ssim_tex_depth = args.lambda_ssim_tex * loss_ssim_tex_depth
        loss_dic['ssim_tex_depth'] = loss_ssim_tex_depth
        ssim_inrgb_depth = pytorch_ssim.ssim(maskRGBs, outputs['re_depth'].unsqueeze(1).repeat(1,3,1,1))
        loss_ssim_inrgb_depth = 1 - ssim_inrgb_depth
        loss_ssim_inrgb_depth = args.lambda_ssim_tex * loss_ssim_inrgb_depth
        loss_dic['ssim_inrgb_depth'] = loss_ssim_inrgb_depth
    else:
        loss_dic['texture'] = torch.zeros(1)
        loss_dic['mrgb'] = torch.zeros(1)
    
    #import pdb; pdb.set_trace()
    # silhouette loss
    if 're_sil' in outputs and 'segms_gt' in examples:
        crit = nn.L1Loss()
        sil_loss = crit(outputs['re_sil'], examples['segms_gt'].float())
        loss_dic['sil'] = sil_loss
    else:
        loss_dic['sil'] = torch.zeros(1)

    # perceptual loss
    if 'perc_features' in outputs and ('texture_con' in examples):
        perc_features = outputs['perc_features']
        batch_size = perc_features[0].shape[0]
        #import pdb; pdb.set_trace()
        loss_percep_batch = torch.mean(torch.abs(perc_features[0]-perc_features[2]),1)+torch.mean(torch.abs(perc_features[1]-perc_features[3]).reshape(batch_size,-1),1)
        loss_percep = torch.sum(loss_percep_batch.mul( examples['texture_con']**2))/torch.sum(( examples['texture_con']**2))
        loss_percep = args.lambda_percep * loss_percep
        loss_dic['loss_percep'] = loss_percep
    else:
        loss_dic['loss_percep'] = torch.zeros(1)
    
    # mesh laplacian loss
    if 'faces' in outputs and 'vertices' in outputs:
        triangle_loss_fn = LaplacianLoss(torch.autograd.Variable(outputs['faces'][0]).cpu(),outputs['vertices'][0])
        triangle_loss = triangle_loss_fn(outputs['vertices'])
        triangle_loss = args.lambda_laplacian * triangle_loss
        loss_dic['triangle'] = triangle_loss
    else:
        loss_dic['triangle'] = torch.zeros(1)

    # mean shape loss
    if 'shape' in outputs:
        shape_loss = torch_f.mse_loss(outputs['shape'], torch.zeros_like(outputs['shape']).to(outputs['shape'].device))
        shape_loss = args.lambda_shape * shape_loss
        loss_dic['mshape'] = shape_loss
    else:
        loss_dic['mshape'] = torch.zeros(1)
    
    return loss_dic


def orthographic_proj_withz(X, trans, scale, offset_z=0.):
    """
    X: B x N x 3
    trans: B x 2: [tx, ty]
    scale: B x 1: [sc]
    Orth preserving the z.
    """
    scale = scale.contiguous().view(-1, 1, 1)
    trans = trans.contiguous().view(scale.size(0), 1, -1)

    proj = scale * X

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)



def trans_proj(outputs, Ks_this, dat_name, xyz_pred_list, verts_pred_list, is_ortho=False):
    #xyz_pred_list, verts_pred_list = list(), list()
    if 'joints' in outputs:
        if dat_name == 'FreiHand':
            output_joints = Mano2Frei(outputs['joints'])
            outputs['joints'] = output_joints
        elif dat_name == 'RHD':
            output_joints = Mano2Frei(outputs['joints'])
            outputs['joints'] = output_joints
        elif dat_name == 'HO3D':
            output_joints = Mano2Frei(outputs['joints'])
            outputs['joints'] = output_joints
        
        if 'joint_2d' in outputs:
            outputs['j2d'] = Mano2Frei(outputs['joint_2d'])
        #import pdb; pdb.set_trace()
        if 'j2d' not in outputs:
            if is_ortho:
                proj_joints = orthographic_proj_withz(outputs['joints'], outputs['trans'], outputs['scale'])
                outputs['j2d'] = proj_joints[:, :, :2]
            else:    
                outputs['j2d'] = proj_func(output_joints, Ks_this)
        #del output_joints
        for i in range(outputs['joints'].shape[0]):
            #import pdb; pdb.set_trace()
            if dat_name == "FreiHand":
                xyz_pred_list.append(outputs['joints'][i].cpu().detach().numpy())
            elif dat_name == "HO3D":
                output_joints_ho3d = Frei2HO3D(outputs['joints'])
                #import pdb; pdb.set_trace()
                output_joints_ho3d = output_joints_ho3d.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
                xyz_pred_list.append(output_joints_ho3d[i].cpu().detach().numpy())
            if 'vertices' in outputs:
                verts_pred_list.append(outputs['vertices'][i].cpu().detach().numpy())
    
    return outputs, xyz_pred_list, verts_pred_list

def save_2d_result(j2d_pred_ED_list,j2d_proj_ED_list,j2d_detect_ED_list,args,j2d_pred_list=[], j2d_proj_list=[], j2d_gt_list=[], j2d_detect_list=[], j2d_detect_con_list=[], epoch=0):
    save_dir = os.path.join(args.base_output_dir,'joint2d_result',str(epoch))
    os.makedirs(save_dir, exist_ok=True)

    j2d_pred_ED = np.asarray(j2d_pred_ED_list)
    j2d_proj_ED = np.asarray(j2d_proj_ED_list)
    j2d_detect_ED = np.asarray(j2d_detect_ED_list)
    print("Prediction - Per Joint Mean Error:",np.mean(j2d_pred_ED,0))
    print("Projection - Per Joint Mean Error:",np.mean(j2d_proj_ED,0))
    print("Detection - Per Joint Mean Error:",np.mean(j2d_detect_ED,0))
    print("Prediction - Overall Mean Error:",np.mean(j2d_pred_ED))
    print("Projection - Overall Mean Error:",np.mean(j2d_proj_ED))
    print("Detection - Overall Mean Error:",np.mean(j2d_detect_ED))
    
    # draw 2d error bar and curves
    eval_errs = [j2d_pred_ED,j2d_proj_ED,j2d_detect_ED]
    eval_names = ['Predicted', 'Projected','Detected']
    metric_type = 'joint'#'max-frame','mean-frame','joint'
    fig = plt.figure(figsize=(16, 6))
    plt.figure(fig.number)
    #draw_error_bar(dataset, eval_errs, eval_names, fig)
    draw_2d_error_curve(eval_errs, eval_names, metric_type, fig)
    #plt.savefig(os.path.join(save_dir,'figures/{}_error.png'.format()))
    plt.savefig(os.path.join(save_dir,'error-pro_{0:.3f}-pre_{1:.3f}-detect_{2:.3f}.png'.format(np.mean(j2d_proj_ED),np.mean(j2d_pred_ED),np.mean(j2d_detect_ED))))
    print('save 2d error image')
    # save error to .txt
    
    savestr=os.path.join(save_dir, 'j2d_proj_ED.txt')
    with open(savestr,'w') as fp:
        for line in j2d_proj_ED_list:
            for l in line:
                fp.write(str(l)+' ')
            fp.write('\n')
    savestr=os.path.join(save_dir, 'j2d_pred_ED.txt')
    with open(savestr,'w') as fp:
        for line in j2d_pred_ED_list:
            for l in line:
                fp.write(str(l)+' ')
            fp.write('\n')
    savestr=os.path.join(save_dir, 'j2d_detect_ED.txt')
    with open(savestr,'w') as fp:
        for line in j2d_detect_ED_list:
            for l in line:
                fp.write(str(l)+' ')
            fp.write('\n')
    j2d_lists = [j2d_pred_list, j2d_proj_list, j2d_gt_list, j2d_detect_list, j2d_detect_con_list]
    j2d_lists_names = ['j2d_pred_list.txt','j2d_proj_list.txt','j2d_gt_list.txt','j2d_detect_list.txt','j2d_detect_con_list.txt']
    for ii in range(len(j2d_lists)):
        if len(j2d_lists[ii])>0:
            savestr=os.path.join(save_dir, j2d_lists_names[ii])
            with open(savestr,'w') as fp:
                for line in j2d_lists[ii]:
                    for l in line:
                        fp.write(str(l)+' ')
                    fp.write('\n')
    print("Write 2D Joints Error at:",save_dir)

def save_2d(examples, outputs, epoch, j2d_pred_ED_list, j2d_proj_ED_list, j2d_detect_ED_list, args):
    #save_dir = os.path.join(args.base_output_dir,'joint2d_result',str(epoch))
    #os.makedirs(save_dir, exist_ok=True)
    if 'j2d_gt' in examples and 'hm_j2d_list' in outputs:
        pred_ED = torch.sqrt(torch.sum((examples['j2d_gt']-outputs['hm_j2d_list'][-1])**2,2))#[8,21]
        j2d_pred_ED_list += pred_ED.cpu().detach().numpy().tolist()
    if 'j2d_gt' in examples and 'j2d' in outputs:
        proj_ED = torch.sqrt(torch.sum((examples['j2d_gt']-outputs['j2d'])**2,2))#[8,21]
        j2d_proj_ED_list += proj_ED.cpu().detach().numpy().tolist()
    if 'j2d_gt' in examples and 'open_2dj' in examples:
        detect_ED = torch.sqrt(torch.sum((examples['j2d_gt']-examples['open_2dj'])**2,2))#[8,21]
        j2d_detect_ED_list += detect_ED.cpu().detach().numpy().tolist()
    return j2d_pred_ED_list, j2d_proj_ED_list, j2d_detect_ED_list

def save_3d(examples, outputs, j3d_ED_list, j2d_ED_list):
    if 'joints' in examples and 'joints' in outputs:
        j3d_ED = torch.sqrt(torch.sum((examples['joints']-outputs['joints'])**2,2))#[8,21]
        j3d_ED_list += j3d_ED.cpu().detach().numpy().tolist()
    if 'j2d_gt' in examples and 'j2d' in outputs:
        j2d_ED = torch.sqrt(torch.sum((examples['j2d_gt']-outputs['j2d'])**2,2))#[8,21]
        j2d_ED_list += j2d_ED.cpu().detach().numpy().tolist()
    return j3d_ED_list, j2d_ED_list

def log_3d_results(j3d_ED_list, j2d_ED_list, epoch, logging):
    j3d_ED = np.asarray(j3d_ED_list)
    j3d_per_joint = np.mean(j3d_ED,0)#[21]
    j3d_mean =np.mean(j3d_ED)#[1]
    j2d_ED = np.asarray(j2d_ED_list)
    j2d_per_joint = np.mean(j2d_ED,0)#[21]
    j2d_mean =np.mean(j2d_ED)#[1]
    #logging.info("Epoch_{0}, Mean_j3d_error:{1}, Mean_j2d_error:{3}, Mean_per_j3d_error:{2}, Mean_per_j2d_error:{4}".format(epoch,j3d_mean,j3d_per_joint,j2d_mean,j2d_per_joint))
    logging.info("Epoch_{0}, Mean_j3d_error:{1}, Mean_j2d_error:{2}".format(epoch, j3d_mean, j2d_mean))


def visualize(mode_train,dat_name,epoch,idx_this,outputs,examples,args, op_outputs=None, writer=None, writer_tag='not-sure', is_val=False):
    # save images
    if mode_train:
        obj_output = os.path.join(args.obj_output,'train')
        image_output = os.path.join(args.image_output, 'train')
    elif is_val:
        obj_output = os.path.join(args.obj_output,'val')
        image_output = os.path.join(args.image_output, 'val')
    else:
        obj_output = os.path.join(args.obj_output,'test')
        image_output = os.path.join(args.image_output, 'test')
    os.makedirs(image_output, exist_ok=True)
    os.makedirs(obj_output, exist_ok=True)
    if mode_train:
        if idx_this % args.demo_freq == 0:
            with torch.no_grad():
                util.displadic(obj_output, image_output, epoch, idx_this, examples, outputs, dat_name, op_outputs=op_outputs, writer=writer, writer_tag=writer_tag, img_wise_save=args.img_wise_save)
    else:
        if idx_this % args.demo_freq_evaluation == 0:
            with torch.no_grad():
                util.displadic(obj_output, image_output, epoch, idx_this, examples, outputs, dat_name, op_outputs=op_outputs, writer=writer, writer_tag=writer_tag, img_wise_save=args.img_wise_save)
            if args.img_wise_save:
                util.multiview_render(image_output, outputs, epoch, idx_this)
                if op_outputs is not None:
                    op_outputs['faces'] = outputs['faces']
                    op_outputs['face_textures'] = outputs['face_textures']
                    op_outputs['render'] = outputs['render']
                    image_output = os.path.join(args.image_output, 'test-op')
                    os.makedirs(image_output, exist_ok=True)
                    util.multiview_render(image_output, outputs, epoch, idx_this)
    #import pdb; pdb.set_trace()
    return 0
    
def save_model(model,optimizer,epoch,current_epoch, args):
    state = {
        'args': args,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + current_epoch,
        #'core': model.core.state_dict(),
    }
    if args.task == 'segm_train':
        state['seghandnet'] = model.module.seghandnet.state_dict()
        save_file = os.path.join(args.state_output, 'seghandnet_{epoch}.t7'.format(epoch=epoch + current_epoch))
        print("Save model at:", save_file)
        torch.save(state, save_file)
    elif args.task == 'train':
        if hasattr(model.module,'encoder'):
            state['encoder'] = model.module.encoder.state_dict()
        if hasattr(model.module,'hand_decoder'):
            state['decoder'] = model.module.hand_decoder.state_dict()
        if hasattr(model.module,'heatmap_attention'):
            state['heatmap_attention'] = model.module.heatmap_attention.state_dict()
        if hasattr(model.module,'rgb2hm'):
            state['rgb2hm'] = model.module.rgb2hm.state_dict()
        if hasattr(model.module,'hm2hand'):
            state['hm2hand'] = model.module.hm2hand.state_dict()
        if hasattr(model.module,'mesh2pose'):
            state['mesh2pose'] = model.module.mesh2pose.state_dict()

        if hasattr(model.module,'percep_encoder'):
            state['percep_encoder'] = model.module.percep_encoder.state_dict()
        
        if hasattr(model.module,'texture_light_from_low'):
            state['texture_light_from_low'] = model.module.texture_light_from_low.state_dict()

        if 'textures' in args.train_requires:
            if hasattr(model.module,'renderer'):
                state['renderer'] = model.module.renderer.state_dict()
            if hasattr(model.module,'texture_estimator'):
                state['texture_estimator'] = model.module.texture_estimator.state_dict()
            if hasattr(model.module,'pca_texture_estimator'):
                state['pca_texture_estimator'] = model.module.pca_texture_estimator.state_dict()
        if 'lights' in args.train_requires:
            if hasattr(model.module,'light_estimator'):
                state['light_estimator'] = model.module.light_estimator.state_dict()
                print("save light estimator")
        save_file = os.path.join(args.state_output, 'texturehand_{epoch}.t7'.format(epoch=epoch + current_epoch))
        print("Save model at:", save_file)
        torch.save(state, save_file)
    elif args.task == 'hm_train':
        state['rgb2hm'] = model.module.rgb2hm.state_dict()
        save_file = os.path.join(args.state_output, 'handhm_{epoch}.t7'.format(epoch=epoch + current_epoch))
        print("Save model at:", save_file)
        torch.save(state, save_file)
    elif args.task == '2Dto3D':
        state['pose_lift_net'] = model.module.pose_lift_net.state_dict()
        save_file = os.path.join(args.state_output, 'pose_lift_net_{epoch}.t7'.format(epoch=epoch + current_epoch))
        print("Save model at:", save_file)
        torch.save(state, save_file)

    return

def write_to_tb(mode_train, writer,loss_dic, epoch, lr=None, is_val=False):
    if mode_train:
        writer.add_scalar('Learning rate', lr, epoch)
        for loss_key in loss_dic:
            if loss_dic[loss_key]>0:
                writer.add_scalar('Train_'+loss_key, loss_dic[loss_key].cpu().detach().numpy(), epoch)
    elif is_val:
        for loss_key in loss_dic:
            if loss_dic[loss_key]>0:
                writer.add_scalar('Val_'+loss_key, loss_dic[loss_key].cpu().detach().numpy(), epoch)
    else:
        for loss_key in loss_dic:
            if loss_dic[loss_key]>0:
                writer.add_scalar('Test_'+loss_key, loss_dic[loss_key].cpu().detach().numpy(), epoch)
    return 0


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]
    #import pdb; pdb.set_trace()
    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def mano_fitting(outputs,Ks=None, op_xyz_pred_list=[], op_verts_pred_list=[], dat_name='FreiHand',args=None):
    # 'pose', 'shape', 'scale','trans', 'rot'
    mano_shape = outputs['shape'].detach().clone()
    mano_pose = outputs['pose'].detach().clone()
    mano_trans = outputs['trans'].detach().clone()
    mano_scale = outputs['scale'].detach().clone()
    mano_rot = outputs['rot'].detach().clone()
    mano_shape.requires_grad = True
    mano_pose.requires_grad = True
    mano_trans.requires_grad = True
    mano_scale.requires_grad = True
    mano_rot.requires_grad = True
    mano_opt_params = [mano_shape, mano_pose,mano_trans,mano_scale,mano_rot]
    
    j2d_2dbranch = outputs['hm_j2d_list'][-1].detach().clone()#[b,21,2]
    j2d_2dbranch_con = torch.ones([j2d_2dbranch.shape[0],j2d_2dbranch.shape[1],1]).to(j2d_2dbranch.device)
    crit_l1 = nn.L1Loss()
    #import pdb;pdb.set_trace()
    iter_total = 151
    batch_time = AverageMeter()
    end = time.time()
    for idx in range(iter_total):
        if idx < 51:
            mano_optimizer = optim.Adam(mano_opt_params, lr=0.01, betas=(0.9, 0.999))
        elif idx < 101:
            mano_optimizer = optim.Adam(mano_opt_params, lr=0.005, betas=(0.9, 0.999))
        else:
            mano_optimizer = optim.Adam(mano_opt_params, lr=0.0025, betas=(0.9, 0.999))
        jv, faces, tsa_poses = rot_pose_beta_to_mesh(mano_rot, mano_pose, mano_shape)
        jv_ts = mano_trans.unsqueeze(1) + torch.abs(mano_scale.unsqueeze(2)) * jv[:,:,:]
        op_joints = jv_ts[:,0:21]
        op_verts = jv_ts[:,21:]
        
        if dat_name == 'FreiHand':
            op_joints = Mano2Frei(op_joints)
        loss = torch.zeros(1)
        # 2dj loss
        j2d = proj_func(op_joints, Ks)
        #import pdb;pdb.set_trace()
        #reprojection_error = gmof(j2d - open_2dj, 100)
        # reprojection loss
        reprojection_distance = torch.sqrt(torch.sum((j2d_2dbranch-j2d)**2,2))
        #reprojection_distance = torch.where(reprojection_distance<5, reprojection_distance**2/10,reprojection_distance-2.5)
        reprojection_loss = args.lambda_j2d * torch.mean(reprojection_distance)
        # bone length loss
        op_bone_direc_loss = bone_direction_loss(j2d, j2d_2dbranch, j2d_2dbranch_con)
        op_bone_direc_loss = args.lambda_bone_direc * op_bone_direc_loss * 0.2

        # pose prior loss
        op_pose_loss = tsa_pose_loss(tsa_poses)
        op_pose_loss = args.lambda_pose * op_pose_loss * 3
        # shape prior loss
        op_shape_loss = torch_f.mse_loss(mano_shape, torch.zeros_like(mano_shape))
        op_shape_loss = args.lambda_shape * op_shape_loss
        # scale prior loss
        out_bone_length = torch.sqrt(torch.sum((op_joints[:,9, :] - op_joints[:,10, :])**2,1))
        op_scale_loss = crit_l1(out_bone_length,torch.ones_like(out_bone_length)*0.0282)
        op_scale_loss = args.lambda_mscale * op_scale_loss
        #import pdb; pdb.set_trace()
        # triangle loss
        triangle_loss_fn = LaplacianLoss(torch.autograd.Variable(outputs['faces'][0]).cpu(),outputs['vertices'][0])
        triangle_loss = triangle_loss_fn(op_verts)
        triangle_loss = args.lambda_laplacian * triangle_loss
        #op_scale_loss = torch.zeros(1)
        #import pdb;pdb.set_trace()
        total_loss = reprojection_loss + op_bone_direc_loss + op_pose_loss + op_shape_loss + op_scale_loss
        
        mano_optimizer.zero_grad()
        total_loss.backward()
        mano_optimizer.step()
        batch_time.update(time.time() - end)
        '''
        if idx%10==0:
            print('Iter: [{0}/{1}]\t' 'loss: {2:.4f}\t' 'Time {batch_time.val:.3f}\t'.format(idx,iter_total,total_loss.data.item(),batch_time=batch_time))
            #print("loss: {:.4f}".format(total_loss.data.item()))
            print("re2dj_loss: {0:.4f}; bone_dire_loss:{1:.4f}; pose_loss:{2:.8f}; shape_loss:{3:.8f}; scale_loss:{4:.8f}".format(reprojection_loss.data.item(),op_bone_direc_loss.data.item(), op_pose_loss.data.item(),op_shape_loss.data.item(),op_scale_loss.data.item()))
        '''
    op_outputs = {}
    op_outputs['j2d'] = j2d
    op_outputs['joints'] = op_joints
    op_outputs['vertices'] = op_verts
    #import pdb; pdb.set_trace()
    if 'render' in outputs:
        op_re_img,op_re_depth,op_re_sil = outputs['render'](op_outputs['vertices'], outputs['faces'], torch.tanh(outputs['face_textures']), mode=None)
    else:
        op_re_img,op_re_depth,op_re_sil = None, None, None
    op_outputs['re_img'], op_outputs['re_deoth'], op_outputs['re_sil'] = op_re_img,op_re_depth,op_re_sil
    for i in range(op_outputs['joints'].shape[0]):
        #import pdb; pdb.set_trace()
        if dat_name == "FreiHand":
            op_xyz_pred_list.append(op_outputs['joints'][i].cpu().detach().numpy())
        elif dat_name == "HO3D":
            output_joints_ho3d = Frei2HO3D(op_outputs['joints'])
            #import pdb; pdb.set_trace()
            output_joints_ho3d = output_joints_ho3d.mul(torch.tensor([1,-1,-1]).view(1,1,-1).float().cuda())
            op_xyz_pred_list.append(output_joints_ho3d[i].cpu().detach().numpy())
        if 'vertices' in outputs:
            op_verts_pred_list.append(outputs['vertices'][i].cpu().detach().numpy())
    return op_outputs, op_xyz_pred_list, op_verts_pred_list