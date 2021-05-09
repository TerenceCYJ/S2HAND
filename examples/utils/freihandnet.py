from utils.hand_model import pose_hand
import torch
from torch import nn
from torch.autograd import Variable
from utils.hand_3d_model import rot_pose_beta_to_mesh
from utils.net_hg import Residual

class FreiPoseHand(nn.Module):
    def __init__(
        self,
        ncomps=6,
        inp_neurons=1536,
        use_pca=True,
        dropout=0,
        ):
        super(FreiPoseHand, self).__init__()
        
        # Base layers
        base_layers = []
        base_layers.append(nn.Linear(inp_neurons, 1024))
        base_layers.append(nn.BatchNorm1d(1024))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Linear(1024, 512))
        base_layers.append(nn.BatchNorm1d(512))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Linear(512, 256))
        base_layers.append(nn.BatchNorm1d(256))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Linear(256, 61))
        self.mano_regressor = nn.Sequential(*base_layers)
    
    def forward(self, features, K):
        mano = self.mano_regressor(features)
        mano = mano.cpu().detach().numpy()
        joints, verts, faces = pose_hand(mano, K)
        #import pdb; pdb.set_trace()
        #joints,  verts, faces = None, None, None
        return verts, faces, joints, poses, shapes

class PoseLiftNet(nn.Module):
    def __init__(self):
        super(PoseLiftNet, self).__init__()
        # Base layers
        base_layers = []
        base_layers.append(nn.Linear(42, 1024))
        base_layers.append(nn.BatchNorm1d(1024))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Dropout())
        base_layers.append(nn.Linear(1024, 512))
        base_layers.append(nn.BatchNorm1d(512))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Dropout())
        base_layers.append(nn.Linear(512, 256))
        base_layers.append(nn.BatchNorm1d(256))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Dropout())
        base_layers.append(nn.Linear(256, 63))
        self.net = nn.Sequential(*base_layers)
    def forward(self, j2d):
        #import pdb;pdb.set_trace()
        batchsize = j2d.shape[0]
        j2d = j2d.reshape(batchsize,-1)
        j3d = self.net(j2d)
        j3d = j3d.reshape(batchsize,-1,3)
        return j3d

class mesh2poseNet(nn.Module):
    def __init__(self):
        super(mesh2poseNet, self).__init__()
        base_layers = []
        base_layers.append(nn.Linear(778*3, 256))
        base_layers.append(nn.BatchNorm1d(256))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Dropout())
        base_layers.append(nn.Linear(256, 128))
        base_layers.append(nn.BatchNorm1d(128))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Dropout())
        base_layers.append(nn.Linear(128, 63))
        self.net = nn.Sequential(*base_layers)
        self.init_weights()
    def init_weights(self):
        #import pdb;pdb.set_trace()
        normal_init(self.net[0],std=0.0001)
        #normal_init(self.net[1],std=0.0001)
        normal_init(self.net[4],std=0.0001)
        #normal_init(self.net[5],std=0.0001)
        normal_init(self.net[8],std=0.0001)

    def forward(self, mesh_coords):
        #import pdb;pdb.set_trace()
        batchsize = mesh_coords.shape[0]
        mesh_coords = mesh_coords.reshape(batchsize,-1)
        j3d = self.net(mesh_coords)
        j3d = j3d.reshape(batchsize,-1,3)
        return j3d

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class MyPoseHand(nn.Module):
    def __init__(
        self,
        ncomps=6,
        inp_neurons=1536,
        use_pca=True,
        dropout=0,
        use_mean_shape = False,
        ):
        super(MyPoseHand, self).__init__()
        self.use_mean_shape = use_mean_shape
        #import pdb;pdb.set_trace()
        # Base layers
        base_layers = []
        base_layers.append(nn.Linear(inp_neurons, 1024))
        base_layers.append(nn.BatchNorm1d(1024))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Linear(1024, 512))
        base_layers.append(nn.BatchNorm1d(512))
        base_layers.append(nn.ReLU())
        self.base_layers = nn.Sequential(*base_layers)

        # Pose Layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 30))#6
        self.pose_reg = nn.Sequential(*layers)

        # Shape Layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 10))
        self.shape_reg = nn.Sequential(*layers)

        # Trans layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 3))
        self.trans_reg = nn.Sequential(*layers)

        # rot layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 3))
        self.rot_reg = nn.Sequential(*layers)

        # scale layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 1))
        #layers.append(nn.ReLU())
        self.scale_reg = nn.Sequential(*layers)

        self.init_weights()
        #self.mean = torch.zeros()
        #self.mean = Variable(torch.FloatTensor([400,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]).cuda())
    def init_weights(self):
        #import pdb; pdb.set_trace()
        #len_scale_reg = len(self.scale_reg)
        '''
        for m in self.scale_reg:
            #import pdb; pdb.set_trace()
            if hasattr(m, 'weight'):#remove ReLU 
                normal_init(m, std=0.1,bias=0.95)
        '''
        normal_init(self.scale_reg[0],std=0.001)
        normal_init(self.scale_reg[2],std=0.001)
        normal_init(self.scale_reg[3],std=0.001,bias=0.95)

        normal_init(self.trans_reg[0],std=0.001)
        normal_init(self.trans_reg[2],std=0.001)
        normal_init(self.trans_reg[3],std=0.001)
        nn.init.constant_(self.trans_reg[3].bias[2],0.65)
        

    def forward(self, features):
        #import pdb; pdb.set_trace()
        base_features = self.base_layers(features)
        theta = self.pose_reg(base_features)#pose
        beta = self.shape_reg(base_features)#shape
        scale = self.scale_reg(base_features)
        trans = self.trans_reg(base_features)
        rot = self.rot_reg(base_features)
        '''
        mano = self.mano_regressor(features)
        mano = mano + mano.mul(self.mean.repeat(mano.shape[0],1).to(device=mano.device))
        scale = mano[:,0]
        trans = mano[:,1:4]
        rot = mano[:,4:7]
        theta = mano[:,7:13]#pose
        beta = mano[:,13:]
        '''
        if self.use_mean_shape:
            beta = torch.zeros_like(beta).to(beta.device)
        # try to set theta as zero tensor
        #theta = torch.zeros_like(theta)#
        #import pdb; pdb.set_trace()
        jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)#rotation pose shape
        #import pdb; pdb.set_trace()
        jv_ts = trans.unsqueeze(1) + torch.abs(scale.unsqueeze(2)) * jv[:,:,:]
        #jv_ts = jv_ts.view(x.size(0),-1) 
        joints = jv_ts[:,0:21]
        verts = jv_ts[:,21:]
        #import pdb; pdb.set_trace()
        #joints,  verts, faces = pose_hand(mano, K)
        #joints,  verts, faces = None, None, None
        #return joints, verts, faces, theta, beta
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses


# like hmr/cmr
class PoseHand(nn.Module):
    def __init__(
        self,
        inp_neurons=1536,
        use_mean_shape = False,
        trans_dim =2,
        ):
        super(PoseHand, self).__init__()
        self.use_mean_shape = use_mean_shape
        #import pdb;pdb.set_trace()
        # Base layers
        base_layers = []
        base_layers.append(nn.Linear(inp_neurons, 1024))
        base_layers.append(nn.BatchNorm1d(1024))
        base_layers.append(nn.ReLU())
        base_layers.append(nn.Linear(1024, 512))
        base_layers.append(nn.BatchNorm1d(512))
        base_layers.append(nn.ReLU())
        self.base_layers = nn.Sequential(*base_layers)

        # Pose Layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 30))#6
        self.pose_reg = nn.Sequential(*layers)

        # Shape Layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 10))
        self.shape_reg = nn.Sequential(*layers)

        # Trans layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, trans_dim))
        self.trans_reg = nn.Sequential(*layers)

        # rot layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 3))
        self.rot_reg = nn.Sequential(*layers)

        # scale layers
        layers = []
        layers.append(nn.Linear(512, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Linear(32, 1))
        #layers.append(nn.ReLU())
        self.scale_reg = nn.Sequential(*layers)

        self.init_weights()
        #self.mean = torch.zeros()
        #self.mean = Variable(torch.FloatTensor([400,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]).cuda())
    def init_weights(self):
        #import pdb; pdb.set_trace()
        #len_scale_reg = len(self.scale_reg)
        '''
        for m in self.scale_reg:
            #import pdb; pdb.set_trace()
            if hasattr(m, 'weight'):#remove ReLU 
                normal_init(m, std=0.1,bias=0.95)
        '''
        normal_init(self.scale_reg[0],std=0.001)
        normal_init(self.scale_reg[2],std=0.001)
        #normal_init(self.scale_reg[3],std=0.001,bias=0.95)
        normal_init(self.scale_reg[3],std=0.001,bias=1)


        normal_init(self.trans_reg[0],std=0.001)
        normal_init(self.trans_reg[2],std=0.001)
        normal_init(self.trans_reg[3],std=0.001)
        nn.init.constant_(self.trans_reg[3].bias,0)
        #import pdb; pdb.set_trace()
        #nn.init.constant_(self.trans_reg[3].bias[1],0.65)
        

    def forward(self, features):
        #import pdb; pdb.set_trace()
        base_features = self.base_layers(features)
        theta = self.pose_reg(base_features)#pose
        beta = self.shape_reg(base_features)#shape
        scale = self.scale_reg(base_features)
        trans = self.trans_reg(base_features)
        rot = self.rot_reg(base_features)
        '''
        mano = self.mano_regressor(features)
        mano = mano + mano.mul(self.mean.repeat(mano.shape[0],1).to(device=mano.device))
        scale = mano[:,0]
        trans = mano[:,1:4]
        rot = mano[:,4:7]
        theta = mano[:,7:13]#pose
        beta = mano[:,13:]
        '''
        if self.use_mean_shape:
            beta = torch.zeros_like(beta).to(beta.device)
        # try to set theta as zero tensor
        #theta = torch.zeros_like(theta)#
        #import pdb; pdb.set_trace()
        jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)#rotation pose shape
        #import pdb; pdb.set_trace()
        verts = jv[:,21:]
        joints = jv[:,0:21]

        #return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses


def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3                                                    
    pose_matrix, _ = rodrigues(poses[:,1:,:].contiguous().view(-1,3))
    #pose_matrix, _ = rodrigues(poses.view(-1,3))    
    pose_matrix = pose_matrix - Variable(torch.from_numpy(np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0),bsize*(keypoints_num-1),axis=0)).cuda())
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix

class Net_HM_Feat(nn.Module):
    def __init__(self, num_heatmap_chan, num_feat_chan, size_input_feature=(64, 64)):
        super(Net_HM_Feat, self).__init__()

        self.num_heatmap_chan = num_heatmap_chan
        self.num_feat_chan = num_feat_chan
        self.size_input_feature = size_input_feature
        self.nRegBlock = 4
        self.nRegModules = 2
        self.heatmap_conv = nn.Conv2d(self.num_heatmap_chan, self.num_feat_chan,
                                      bias=True, kernel_size=1, stride=1)
        self.encoding_conv = nn.Conv2d(self.num_feat_chan, self.num_feat_chan,
                                       bias=True, kernel_size=1, stride=1)
        _reg_ = []
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                _reg_.append(Residual(self.num_feat_chan, self.num_feat_chan))
        self.reg_ = nn.ModuleList(_reg_)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_scale = 2 ** self.nRegBlock

        # fc layers
        self.num_feat_out = self.num_feat_chan * (
                    size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2))

    def forward(self, hm_list, encoding_list):
        x = self.heatmap_conv(hm_list[-1]) + self.encoding_conv(encoding_list[-1])
        if len(encoding_list) > 1:
            x = x + encoding_list[-2]
        
        # x: B x num_feat_chan x 64 x 64
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                x = self.reg_[i * self.nRegModules + j](x)
            x = self.maxpool(x)
        # x: B x num_feat_chan x 4 x 4
        out = x.view(x.size(0), -1)
        # x: B x 4096
        return out

class Residual_block(nn.Module):
    def __init__(self, numIn, times):
        super(Residual_block, self).__init__()
        self.numIn = numIn
        self.numOut = numIn
        res_list=[]
        for i in range(times):
            res_l = nn.Sequential(
                nn.BatchNorm2d(self.numOut),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.numOut, self.numOut // 2, bias=True, kernel_size=1)
            )
            res_list.append()
            self.numOut = self.numOut // 2

        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

class Net_hm_fuse(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Net_hm_fuse, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan


class HM2Mano(nn.Module):
    def __init__(self, num_heatmap_chan, num_feat_chan, num_mesh_output_chan, graph_L, size_input_feature=(64, 64)):
        super(HM2Mano, self).__init__()

        self.feat_net = Net_HM_Feat(num_heatmap_chan, num_feat_chan, size_input_feature)
        self.hand_decoder = MyPoseHand(inp_neurons=4096)
        #self.mesh_net = Graph_CNN_Feat_Mesh(self.feat_net.num_feat_out, num_mesh_output_chan, graph_L)

    def forward(self, hm_list, encoding_list):
        feat = self.feat_net(hm_list, encoding_list)  # B x 4096
        joints, verts, faces, theta, beta = self.hand_decoder(feat)
        return joints, verts, faces, theta, beta, feat
'''
class HM2Mano(nn.Module):
    def __init__(self, num_heatmap_chan, num_feat_chan, num_mesh_output_chan, graph_L, size_input_feature=(64, 64)):
        super(HM2Mano, self).__init__()

        self.feat_net = Net_HM_Feat(num_heatmap_chan, num_feat_chan, size_input_feature)
        #self.hand_decoder = MyPoseHand(inp_neurons=4096)
        self.hand_decoder = FreiPoseHand(inp_neurons=4096)
        #self.mesh_net = Graph_CNN_Feat_Mesh(self.feat_net.num_feat_out, num_mesh_output_chan, graph_L)

    def forward(self, hm_list, encoding_list, K):
        feat = self.feat_net(hm_list, encoding_list)  # B x 4096
        joints, verts, faces, theta, beta = self.hand_decoder(feat, K)
        return joints, verts, faces, theta, beta, feat
'''

