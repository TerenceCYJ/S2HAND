from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import time
import skimage.io as io
import torch
import matplotlib.pyplot as plt

""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def proj_func(xyz, K):
    '''
    xyz: N x num_points x 3
    K: N x 3 x 3
    '''
    uv = torch.bmm(K,xyz.permute(0,2,1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:,:,:2]).to(device=uv.device)
    out_uv = torch.addcdiv(out_uv, 1 ,uv[:,:,:2], uv[:,:,2].unsqueeze(-1).repeat(1,1,2))
    return out_uv


def backproject_ortho(uv_root, scales, Ks):
    focal = 0.5 * (Ks[:, 0, 0] + Ks[:, 1, 1])#[64]
    uvp = Ks[:,0:2,2].to(uv_root.device)#[64,2]
    
    #uvp = uvp.unsqueeze(1)#[64,1,2]
    uv_root -= uvp
    xyz = torch.cat([uv_root, focal.unsqueeze(1).to(uv_root.device)],dim=1)
    #import pdb; pdb.set_trace()
    #xyz /= scale
    xyz = xyz.mul(scales)
    return xyz




""" Draw functions. """
def plot_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', markersize=1, order='hw', draw_kp=True, dataset_name = 'FreiHand'):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])
    

    colors = colors[:, ::-1]

    # define connections and colors of the bones
    if dataset_name == 'FreiHand':
        bones = [((0, 1), colors[1, :]),
                ((1, 2), colors[2, :]),
                ((2, 3), colors[3, :]),
                ((3, 4), colors[4, :]),

                ((0, 5), colors[5, :]),
                ((5, 6), colors[6, :]),
                ((6, 7), colors[7, :]),
                ((7, 8), colors[8, :]),

                ((0, 9), colors[9, :]),
                ((9, 10), colors[10, :]),
                ((10, 11), colors[11, :]),
                ((11, 12), colors[12, :]),

                ((0, 13), colors[13, :]),
                ((13, 14), colors[14, :]),
                ((14, 15), colors[15, :]),
                ((15, 16), colors[16, :]),

                ((0, 17), colors[17, :]),
                ((17, 18), colors[18, :]),
                ((18, 19), colors[19, :]),
                ((19, 20), colors[20, :])]
        kp_colors = colors
    elif dataset_name == 'RHD':
        import pdb; pdb.set_trace()
        # should transfer RHD to FreiHand index
        bones = [((0, 4), colors[1, :]),
                ((1, 2), colors[2, :]),
                ((2, 3), colors[3, :]),
                ((3, 4), colors[4, :]),

                ((0, 8), colors[5, :]),
                ((5, 6), colors[6, :]),
                ((6, 7), colors[7, :]),
                ((7, 8), colors[8, :]),

                ((0, 12), colors[9, :]),
                ((9, 10), colors[10, :]),
                ((10, 11), colors[11, :]),
                ((11, 12), colors[12, :]),

                ((0, 16), colors[13, :]),
                ((13, 14), colors[14, :]),
                ((14, 15), colors[15, :]),
                ((15, 16), colors[16, :]),

                ((0, 20), colors[17, :]),
                ((17, 18), colors[18, :]),
                ((18, 19), colors[19, :]),
                ((19, 20), colors[20, :])]
        kp_colors = colors
    elif dataset_name == 'Obman' or dataset_name == 'openpose':
        bones = [((0, 1), colors[1, :]),
                ((1, 2), colors[2, :]),
                ((2, 3), colors[3, :]),
                ((3, 4), colors[4, :]),

                ((0, 5), colors[5, :]),
                ((5, 6), colors[6, :]),
                ((6, 7), colors[7, :]),
                ((7, 8), colors[8, :]),

                ((0, 9), colors[9, :]),
                ((9, 10), colors[10, :]),
                ((10, 11), colors[11, :]),
                ((11, 12), colors[12, :]),

                ((0, 13), colors[13, :]),
                ((13, 14), colors[14, :]),
                ((14, 15), colors[15, :]),
                ((15, 16), colors[16, :]),

                ((0, 17), colors[17, :]),
                ((17, 18), colors[18, :]),
                ((18, 19), colors[19, :]),
                ((19, 20), colors[20, :])]
        kp_colors = colors
    elif dataset_name == 'HO3D':
        bones = [((0, 13), colors[1, :]),
                ((13, 14), colors[2, :]),
                ((14, 15), colors[3, :]),
                ((15, 16), colors[4, :]),

                ((0, 1), colors[5, :]),
                ((1, 2), colors[6, :]),
                ((2, 3), colors[7, :]),
                ((3, 17), colors[8, :]),

                ((0, 4), colors[9, :]),
                ((4, 5), colors[10, :]),
                ((5, 6), colors[11, :]),
                ((6, 18), colors[12, :]),

                ((0, 10), colors[13, :]),
                ((10, 11), colors[14, :]),
                ((11, 12), colors[15, :]),
                ((12, 19), colors[16, :]),

                ((0, 7), colors[17, :]),
                ((7, 8), colors[18, :]),
                ((8, 9), colors[19, :]),
                ((9, 20), colors[20, :])]
        kp_colors = [colors[0, :],
                    colors[5, :],
                    colors[6, :],
                    colors[7, :],
                    colors[9, :],
                    colors[10, :],

                    colors[11, :],
                    colors[17, :],
                    colors[18, :],
                    colors[19, :],
                    colors[13, :],

                    colors[14, :],
                    colors[15, :],
                    colors[1, :],
                    colors[2, :],
                    colors[3, :],

                    colors[4, :],
                    colors[8, :],
                    colors[12, :],
                    colors[16, :],
                    colors[20, :]]
    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            #axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])
            #import pdb;pdb.set_trace()
            #axis.plot(coords_hw[i, 1], coords_hw[i, 0], '.', color=kp_colors[i], markersize=markersize)
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], marker='o', color=kp_colors[i], markersize=markersize)

def plot_hand_3d(axis,xyz,vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])
    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]
    #xyz = xyz[:,:, ::-1]
    if vis is None:
        vis = np.ones_like(xyz[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue
        coord1 = xyz[connection[0], :]
        coord2 = xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        #import pdb; pdb.set_trace()
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            #import pdb;pdb.set_trace()
            #axis.scatter([xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]], 'o', color=colors[i, :])
            axis.scatter([xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]], color=colors[i, :])


""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return list(zip(K_list, mano_list, xyz_list))


class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def read_msk(idx, base_path):
    mask_path = os.path.join(base_path, 'training', 'mask',
                             '%08d.jpg' % idx)
    _assert_exist(mask_path)
    return io.imread(mask_path)


def img_cvt(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Mano2Frei(Mano_joints):#[b,21,3]
    #import pdb; pdb.set_trace()
    FreiHand_joints = torch.zeros_like(Mano_joints).to(Mano_joints.device) # init empty list

    # manoId, FreiId
    mapping = {0: 0, #Wrist
               1: 5, 2: 6, 3: 7, 4: 8, #Index
               5: 9, 6: 10, 7: 11, 8: 12, #Middle
               9: 17, 10: 18, 11: 19, 12: 20, # Pinky
               13: 13, 14: 14, 15: 15, 16: 16, # Ring
               17: 1, 18: 2, 19: 3, 20: 4,} # Thumb

    for manoId, myId in mapping.items():
        FreiHand_joints[:,myId] = Mano_joints[:,manoId]
    #import pdb; pdb.set_trace()
    return FreiHand_joints

def Mano2RHD(Mano_joints):#[b,21,3]
    #import pdb; pdb.set_trace()
    rhd_joints = torch.zeros_like(Mano_joints).to(Mano_joints.device) # init empty list

    # manoId, rhdId
    mapping = {0: 0, #Wrist
               1: 8, 2: 7, 3: 6, 4: 5, #Index
               5: 12, 6: 11, 7: 10, 8: 9, #Middle
               9: 20, 10: 19, 11: 18, 12: 17, # Pinky
               13: 16, 14: 15, 15: 14, 16: 13, # Ring
               17: 4, 18: 3, 19: 2, 20: 1,} # Thumb

    for manoId, myId in mapping.items():
        rhd_joints[:,myId,:] = Mano_joints[:,manoId, :]
    #import pdb; pdb.set_trace()
    return rhd_joints

def RHD2Mano(rhd_joints):
    Mano_joints = torch.zeros_like(rhd_joints).to(rhd_joints.device) # init empty list
    # manoId, rhdId
    mapping = {0: 0, #Wrist
               1: 8, 2: 7, 3: 6, 4: 5, #Index
               5: 12, 6: 11, 7: 10, 8: 9, #Middle
               9: 20, 10: 19, 11: 18, 12: 17, # Pinky
               13: 16, 14: 15, 15: 14, 16: 13, # Ring
               17: 4, 18: 3, 19: 2, 20: 1,} # Thumb
    for manoId, myId in mapping.items():
        Mano_joints[:,manoId] = rhd_joints[:,myId]
    #import pdb; pdb.set_trace()
    return Mano_joints

def RHD2Frei(rhd_joints):
    FreiHand_joints = torch.zeros_like(rhd_joints).to(rhd_joints.device) # init empty list
    # myId, rhdId
    mapping = {0: 0, #Wrist
               1: 4, 2: 3, 3: 2, 4: 1, #Index
               5: 8, 6: 7, 7: 6, 8: 5, #Middle
               9: 12, 10: 11, 11: 10, 12: 9, # Pinky
               13: 16, 14: 15, 15: 14, 16: 13, # Ring
               17: 20, 18: 19, 19: 18, 20: 17,} # Thumb
    for myId,rhdId in mapping.items():
        FreiHand_joints[:,myId] = rhd_joints[:,rhdId]
    #import pdb; pdb.set_trace()
    return FreiHand_joints

def HO3D2Frei(ho3d_joints):
    FreiHand_joints = torch.zeros_like(ho3d_joints).to(ho3d_joints.device) # init empty list
    # myId, ho3dId
    mapping = {0: 0, #Wrist
               1: 13, 2: 14, 3: 15, 4: 16, #thumb
               5: 1, 6: 2, 7: 3, 8: 17, # index
               9: 4, 10: 5, 11: 6, 12: 18, # middle
               13: 10, 14: 11, 15: 12, 16: 19, # ring
               17: 7, 18: 8, 19: 9, 20: 20,} # pinky
    for myId,ho3dId in mapping.items():
        FreiHand_joints[:,myId] = ho3d_joints[:,ho3dId]
    #import pdb; pdb.set_trace()
    return FreiHand_joints

def Frei2HO3D(FreiHand_joints):
    ho3d_joints = torch.zeros_like(FreiHand_joints).to(FreiHand_joints.device) # init empty list
    # FreiId, ho3dId
    mapping = {0: 0, #Wrist
               1: 13, 2: 14, 3: 15, 4: 16, #thumb
               5: 1, 6: 2, 7: 3, 8: 17, # index
               9: 4, 10: 5, 11: 6, 12: 18, # middle
               13: 10, 14: 11, 15: 12, 16: 19, # ring
               17: 7, 18: 8, 19: 9, 20: 20,} # pinky
    for FreiId,ho3dId in mapping.items():
        ho3d_joints[:,ho3dId] = FreiHand_joints[:,FreiId]
    return ho3d_joints

def Mano2HO3D(Mano_joints):#[b,21,3]
    #import pdb; pdb.set_trace()
    ho3d_joints = torch.zeros_like(Mano_joints).to(Mano_joints.device) # init empty list
    

    # manoId, HO3DId
    mapping = {0: 0, #Wrist
               1: 1, 2: 2, 3: 3, 4: 17, #Index
               5: 4, 6: 5, 7: 6, 8: 18, #Middle
               9: 7, 10: 8, 11: 9, 12: 20, # Pinky
               13: 10, 14: 11, 15: 12, 16: 19, # Ring
               17: 13, 18: 14, 19: 15, 20: 16,} # Thumb

    for manoId, myId in mapping.items():
        ho3d_joints[:,myId,:] = Mano_joints[:,manoId, :]
    #import pdb; pdb.set_trace()
    return ho3d_joints


def Mano2Obman(Mano_joints):#[b,21,3]
    #import pdb; pdb.set_trace()
    obman_joints = torch.zeros_like(Mano_joints).to(Mano_joints.device) # init empty list

    # manoId, FreiId
    mapping = {0: 0, #Wrist
               1: 5, 2: 6, 3: 7, 4: 8, #Index
               5: 9, 6: 10, 7: 11, 8: 12, #Middle
               9: 17, 10: 18, 11: 19, 12: 20, # Pinky
               13: 13, 14: 14, 15: 15, 16: 16, # Ring
               17: 1, 18: 2, 19: 3, 20: 4,} # Thumb

    for manoId, myId in mapping.items():
        obman_joints[:,myId,:] = Mano_joints[:,manoId, :]
    #import pdb; pdb.set_trace()
    return obman_joints

# openpose mapping
def open2HO3D(open_2dj):#[b,21,3]
    #import pdb; pdb.set_trace()
    ho_2dj = torch.zeros_like(open_2dj).to(open_2dj.device) # init empty list

    # openpose_Id, ho3d_Id
    mapping = {0: 0, #Wrist
               1: 13, 2: 14, 3: 15, 4: 16, #Index
               5: 1, 6: 2, 7: 3, 8: 17, #Middle
               9: 4, 10: 5, 11: 6, 12: 18, # Pinky
               13: 10, 14: 11, 15: 12, 16: 19, # Ring
               17: 7, 18: 8, 19: 9, 20: 20,} # Thumb

    for openId, myId in mapping.items():
        ho_2dj[:,myId,:] = open_2dj[:,openId, :]
    #import pdb; pdb.set_trace()
    return ho_2dj

def batch_depth2pc(gt_depth,Ks,num_pc=1000):#[b,width,hight]
    uv_index = torch.nonzero(gt_depth)#[N,3] [i,u,v] 0<=i<Batch_size
    z_gt = gt_depth[gt_depth>0]#[N]
    cx_gt = Ks[:,0,2].view(-1,1,1).repeat(1,gt_depth.shape[1],gt_depth.shape[1])
    cx_gt = cx_gt[gt_depth>0]
    cy_gt = Ks[:,1,2].view(-1,1,1).repeat(1,gt_depth.shape[1],gt_depth.shape[1])
    cy_gt = cy_gt[gt_depth>0]
    fx_gt = Ks[:,0,0].view(-1,1,1).repeat(1,gt_depth.shape[1],gt_depth.shape[1])
    fx_gt = fx_gt[gt_depth>0]
    fy_gt = Ks[:,1,1].view(-1,1,1).repeat(1,gt_depth.shape[1],gt_depth.shape[1])
    fy_gt = fy_gt[gt_depth>0]
    #import pdb; pdb.set_trace()
    x_gt = ((uv_index[:,1].float()-cx_gt).mul(z_gt)).mul(torch.reciprocal(fx_gt+1e-5))
    y_gt = ((uv_index[:,2].float()-cy_gt).mul(z_gt)).mul(torch.reciprocal(fy_gt+1e-5))
    
    
    batch_size = gt_depth.shape[0]
    xyz = torch.zeros(batch_size,num_pc,3)
    for i in range(batch_size):
        index = torch.nonzero(uv_index[:,0]==i).view(-1)
        #print(i)
        #import pdb;pdb.set_trace()
        if len(index)>1:
            indice = torch.randint(0, index.shape[0]-1, (num_pc,)).to(index.device)
            index = torch.index_select(index, 0, indice)

            #index = torch.index_select(index, 0, indice)
            #import pdb; pdb.set_trace()
            xyz[i] = torch.cat((torch.index_select(x_gt, 0, index).view(-1,1),torch.index_select(y_gt, 0, index).view(-1,1),torch.index_select(z_gt, 0, index).view(-1,1)),1)
            #import pdb;pdb.set_trace()
        #else:
        #    xyz[i] = xyz[]
    return xyz

class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        #keypoint_gt = np.squeeze(keypoint_gt)
        #keypoint_pred = np.squeeze(keypoint_pred)
        #import pdb;pdb.set_trace()
        #keypoint_vis = np.squeeze(keypoint_vis).astype('bool')
        
        assert len(keypoint_gt.shape) == 3
        assert len(keypoint_pred.shape) == 3
        assert len(keypoint_vis.shape) == 2

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        #import pdb;pdb.set_trace()
        euclidean_dist = torch.sqrt(torch.sum(diff**2,2))
        #euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        batch_size = keypoint_gt.shape[0]
        num_kp = keypoint_gt.shape[1]
        
        for i in range(num_kp):
            for j in range(batch_size):
                if keypoint_vis[j][i]:
                    self.data[i].append(euclidean_dist[j][i].cpu().data.item())
        #import pdb;pdb.set_trace()

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds



def calc_auc(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)

    return integral / norm

def get_stb_ref_curves():
    """
        Returns results of various baseline methods on the Stereo Tracking Benchmark Dataset reported by:
        Zhang et al., ‘3d Hand Pose Tracking and Estimation Using Stereo Matching’, 2016
    """
    curve_list = list()
    thresh_mm = np.array([20.0, 25, 30, 35, 40, 45, 50])
    pso_b1 = np.array([0.32236842,  0.53947368,  0.67434211,  0.75657895,  0.80921053, 0.86513158,  0.89473684])
    curve_list.append((thresh_mm, pso_b1, 'PSO (AUC=%.3f)' % calc_auc(thresh_mm, pso_b1)))
    icppso_b1 = np.array([ 0.51973684,  0.64473684,  0.71710526,  0.77302632,  0.80921053, 0.84868421,  0.86842105])
    curve_list.append((thresh_mm, icppso_b1, 'ICPPSO (AUC=%.3f)' % calc_auc(thresh_mm, icppso_b1)))
    chpr_b1 = np.array([ 0.56578947,  0.71710526,  0.82236842,  0.88157895,  0.91447368, 0.9375,  0.96052632])
    curve_list.append((thresh_mm, chpr_b1, 'CHPR (AUC=%.3f)' % calc_auc(thresh_mm, chpr_b1)))
    return curve_list

def plot_curve(curve_list,curve_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for t, v, name in curve_list:
        ax.plot(t, v, label=name)
    ax.set_xlabel('threshold in mm')
    ax.set_ylabel('PCK')
    plt.legend(loc='lower right')
    #plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    print("save image at:", curve_path)

class ObmanEvalUtil:
    """ Util class for evaluation networks.
    """

    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_pred, keypoint_vis=None):
        """
        Used to feed data to the class.
        Stores the euclidean distance between gt and pred, when it is visible.
        """
        if isinstance(keypoint_gt, torch.Tensor):
            keypoint_gt = keypoint_gt.numpy()
        if isinstance(keypoint_pred, torch.Tensor):
            keypoint_pred = keypoint_pred.numpy()
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)

        if keypoint_vis is None:
            keypoint_vis = np.ones_like(keypoint_gt[:, 0])
        keypoint_vis = np.squeeze(keypoint_vis).astype("bool")

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1
        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))#[21]
        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])
        # self.data [21, n] n: number of samples

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)#50

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):#21
            # mean/median error
            mean, median = self._get_epe(part_id)
            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)
        # Display error per keypoint
        epe_mean_joint = epe_mean_all#[21]
        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        '''
        draw_curve = True
        if draw_curve:
            eval_num = len(self.data[0])#number of frames
            joint_num = len(self.data)#21
        '''


        return (
            epe_mean_all,
            epe_mean_joint,
            epe_median_all,
            auc_all,
            pck_curve_all,
            thresholds,
        )

# transfer openpose caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)