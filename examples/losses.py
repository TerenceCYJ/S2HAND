import torch
import torch.nn as nn
import numpy as np

import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    '''
    https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    '''
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

try:
    from efficientnet_pt.model import EfficientNet
except:
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from efficientnet_pt import EfficientNet
class EffiPerceptualLoss(nn.Module):
    def __init__(self, pretrained_model = None, resize=True):
        super(EffiPerceptualLoss, self).__init__()
        if pretrained_model is None:
            self.percep_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.percep_encoder = EfficientNet.from_name('efficientnet-b0')
            state_dict = torch.load(pretrained_model)
            self.percep_encoder.load_state_dict(state_dict)
            print('load efficientnet from:', pretrained_model)

    def forward(self, input, target):
        loss = 0.0
        input_1, input_2 = self.percep_encoder(input)
        target_1, target_2 = self.percep_encoder(target)
        loss += torch.nn.functional.l1_loss(input_1, target_1)
        loss += torch.nn.functional.l1_loss(input_2, target_2)
        return loss
    def extract_features(self, input, target):
        input_1, input_2 = self.percep_encoder(input)
        target_1, target_2 = self.percep_encoder(target)
        return [input_1,input_2,target_1, target_2]

'''
from openpose_detector.src.model import handpose_model
from openpose_detector.src.util import transfer
class OpenPosePerceptualLoss(nn.Module):
    def __init__(self, model_path = None, resize=True):
        super(OpenPosePerceptualLoss, self).__init__()
        self.model = handpose_model()
        model_dict = transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def forward(self, input, input0):
        batch_size = input.shape[0]
        feature = self.model.model1_0(input)
        feature0 = self.model.model1_0(input0)#[b,128,28,28]
        feature_1 = self.model.model1_1(feature)
        feature0_1 = self.model.model1_1(feature0)#[b,22,28,28]
        out_stage2 = self.model.model2(torch.cat([feature_1, feature], 1))
        out0_stage2 = self.model.model2(torch.cat([feature0_1, feature0], 1))#[b,22,28,28]
        out_stage3 = self.model.model3(torch.cat([out_stage2, feature], 1))
        out0_stage3 = self.model.model3(torch.cat([out0_stage2, feature0], 1))#[b,22,28,28]
        out_stage4 = self.model.model4(torch.cat([out_stage3, feature], 1))
        out0_stage4 = self.model.model4(torch.cat([out0_stage3, feature0], 1))
        out_stage5 = self.model.model5(torch.cat([out_stage4, feature], 1))
        out0_stage5 = self.model.model5(torch.cat([out0_stage4, feature0], 1))
        loss = torch.mean(torch.abs(feature-feature0).reshape(batch_size,-1),1)
        loss += torch.mean(torch.abs(feature_1-feature0_1).reshape(batch_size,-1),1)
        loss += torch.mean(torch.abs(out_stage2-out0_stage2).reshape(batch_size,-1),1)
        loss += torch.mean(torch.abs(out_stage3-out0_stage3).reshape(batch_size,-1),1)
        loss += torch.mean(torch.abs(out_stage4-out0_stage4).reshape(batch_size,-1),1)
        loss += torch.mean(torch.abs(out_stage5-out0_stage5).reshape(batch_size,-1),1)

        return loss#[feature, feature0]
'''
def iou(predict, target, eps=1e-6):
    #remove nan
    #predict[predict!= predict] = 0
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target):
    return 1 - iou(predict, target)

'''
def singleview_iou_loss(predicts, targets):
    mean_iou_loss = 0
    for i in range(len(predicts)):
        predict = predicts[i]#[4,224,224]
        target = targets[i]#[3,224,224]
        loss =iou_loss(predict[:3], target)
        mean_iou_loss += loss
    mean_iou_loss = mean_iou_loss/len(predicts)
    #import pdb; pdb.set_trace()
    return mean_iou_loss
'''

def multiview_iou_loss(predicts, targets_a, targets_b):
    loss = (iou_loss(predicts[0][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[1][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[2][:, 3], targets_b[:, 3]) + \
            iou_loss(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss

def image_l1_loss(predicts, targets):
    loss = 3 * torch.mean(torch.abs(predicts - targets))
    return loss

def tsa_pose_loss(tsaposes):
    #tilt-swing-azimuth pose prior loss
    '''
    tsaposes: (B,16,3)
    '''
    pi = np.pi
    '''
    max_nonloss = torch.tensor([[3.15,0.01,0.01],
                                [5*pi/180,10*pi/180,100*pi/180],#0
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#6
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,pi/8,pi/8],#12
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]]).float().to(tsaposes.device)
    min_nonloss = torch.tensor([[3.13,-0.01,-0.01],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-10*pi/180]]).float().to(tsaposes.device)
    '''
    max_nonloss = torch.tensor([[3.15,0.01,0.01],
                                [5*pi/180,10*pi/180,100*pi/180],#0 INDEX
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#3 MIDDLE
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,20*pi/180,100*pi/180],#6 PINKY
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,10*pi/180,100*pi/180],#9 RING
                                [5*pi/180,5*pi/180,100*pi/180],
                                [5*pi/180,5*pi/180,100*pi/180],
                                [90*pi/180,3*pi/16,pi/8],#12 THUMB
                                [5*pi/180,5*pi/180,pi/8],
                                [5*pi/180,5*pi/180,100*pi/180]]).float().to(tsaposes.device)
    min_nonloss = torch.tensor([[3.13,-0.01,-0.01],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#0
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#3
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-20*pi/180,-10*pi/180,-10*pi/180],#6
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-10*pi/180,-10*pi/180],#9
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [-5*pi/180,-5*pi/180,-10*pi/180],
                                [0,-pi/8,-pi/8],#12
                                [-5*pi/180,-5*pi/180,-pi/8],
                                [-5*pi/180,-5*pi/180,-20*pi/180]]).float().to(tsaposes.device)
    median_nonloss = (max_nonloss+min_nonloss)/2
    #tsa_pose_errors = torch.where(tsaposes>max_nonloss.unsqueeze(0),tsaposes-median_nonloss.unsqueeze(0),torch.zeros_like(tsaposes)) + torch.where(tsaposes<min_nonloss.unsqueeze(0),-tsaposes+median_nonloss.unsqueeze(0),torch.zeros_like(tsaposes))
    tsa_pose_errors = torch.where(tsaposes>max_nonloss.unsqueeze(0),tsaposes-max_nonloss.unsqueeze(0),torch.zeros_like(tsaposes)) + torch.where(tsaposes<min_nonloss.unsqueeze(0),-tsaposes+min_nonloss.unsqueeze(0),torch.zeros_like(tsaposes))
    tsa_pose_loss = torch.mean(tsa_pose_errors.mul(torch.tensor([1,1,2]).float().to(tsa_pose_errors.device)))#.cpu()
    #import pdb; pdb.set_trace()
    return tsa_pose_loss

def bone_direction_loss(j2d, open_2dj, open_2dj_con):
    '''
    j2d (b,21,2)
    open_2dj (b,21,2)
    open_2dj_con (b,21,1)
    '''
    device = j2d.device
    batch_size = j2d.shape[0]
    # bone vector
    mat_20_21 = torch.tensor([[-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#0,1
                            [0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1]])
    mat_21_20 = torch.transpose(mat_20_21,0,1).to(j2d.device).float()
    
    bone_vec_batch = torch.bmm(torch.transpose(j2d,1,2),mat_21_20.unsqueeze(0).repeat(j2d.shape[0],1,1))#[b,2,20]
    bone_vec_open_batch = torch.bmm(torch.transpose(open_2dj,1,2),mat_21_20.unsqueeze(0).repeat(j2d.shape[0],1,1))#[b,2,20]
    #bone_len = torch.sqrt(torch.sum(bone_vec_batch**2,1))#[b,20]
    #bone_open_len = torch.sqrt(torch.sum(bone_vec_open_batch**2,1))#[b,20]

    bone_vec_nm = bone_vec_batch / (torch.sqrt(torch.sum(bone_vec_batch**2,1)).unsqueeze(1)+1e-4)#[b,2,20]
    bone_vec_open_nm = bone_vec_open_batch / (torch.sqrt(torch.sum(bone_vec_open_batch**2,1)).unsqueeze(1)+1e-4)#[b,2,20]

    # bone confidence
    confidence_mat_21_21 = torch.tensor([[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],#0,1
                                        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).to(torch.uint8)
    #import pdb; pdb.set_trace()
    confi_matrixs = torch.bmm(open_2dj_con,torch.transpose(open_2dj_con,1,2)).mul(confidence_mat_21_21.unsqueeze(0).to(device).float())
    confs = torch.transpose(confi_matrixs,1,2)[torch.transpose(confidence_mat_21_21.unsqueeze(0).repeat(batch_size,1,1).to(torch.uint8),1,2)]
    confs = confs.view(-1,20)#[b,20]
    
    bone_direction_loss = torch.mean(torch.sum((bone_vec_nm-bone_vec_open_nm)**2,1).mul(confs))
    #import pdb; pdb.set_trace()
    return bone_direction_loss


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins, 1)

        return loss_1, loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = (
            xx[:, diag_ind_x, diag_ind_x]
            .unsqueeze(1)
            .expand_as(zz.transpose(2, 1))
        )
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P