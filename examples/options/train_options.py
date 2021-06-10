import os
import argparse

def get_parser():
    """
    :return: return a parser which stores the arguments for training the network
    """
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--mode', type=list, default=['training'])#['training','evaluation']
    parser.add_argument('--task', type=str, default='train',
                        help='task: train, test, segm_train, hm_train, check, 2Dto3D')
    # For inputs
    parser.add_argument('--train_queries', type=list, default=['images','Ks','joints','open_2dj'])
    # FreiHand
    # - 'images','masks','maskRGBs','Ks','scales','manos','joints','verts','open_2dj','CRFmasks',
    # - 'trans_images','trans_open_2dj','trans_Ks','trans_CRFmasks','trans_masks','trans_joints','trans_verts',
    # RHD
    # - 'trans_images','base_masks','base_Ks','base_joints','base_joints2d','base_depth',
    # - 'trans_images','trans_masks','trans_Ks','joints','joints_normed','trans_joints2d','trans_depth','sides','joints2d_vis',
    # Obman
    # - 'base_images','base_instances','base_segms','base_j3d','base_j2d','base_verts','base_camintrs','maskRGBs',# trans 'trans_images',
    # - 'trans_instances',#Full mask 'trans_segms', 'trans_j3d', 'trans_j2d', 'trans_verts','trans_camintrs','sides','center3d',
    parser.add_argument('--val_queries', type=list, default=['images','Ks','joints','open_2dj'])

    # For encoder
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b3')
    parser.add_argument('--neck_name', type=str, default='wo')#'BiFPN'

    parser.add_argument('--camera_mode', type=str, default='projection')
    parser.add_argument('--perspective', type=bool, default=False)
    

    # For outputs
    parser.add_argument('--train_requires', type=list, default=['joints','verts'])#["heatmaps",'segms',"textures","lights"]
    parser.add_argument('--test_requires', type=list, default=['joints','verts'])

    parser.add_argument('--regress_mode', type=str, default='mano',#check
                        help='hm2mano,mano')
    parser.add_argument('--renderer_mode', type=str, default='NR',
                        help='SoftRas,NR,opendr')
    parser.add_argument('--texture_mode', type=str, default='surf',help='pca, surf, same, nn_same, vert, html')             
    parser.add_argument('--use_mean_shape', type=bool, default=True)#check
    parser.add_argument('--use_2d_as_attention', type=bool, default=False)
    #parser.add_argument('--losses', type=list, default=['tex','mtex'])
    parser.add_argument('--losses', type=list, default=['mscale'])
    parser.add_argument('--losses_frei', type=list, default=[])
    parser.add_argument('--losses_rhd', type=list, default=[])
    parser.add_argument('--losses_init', type=list, default=[])
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--train_init', type=list, default=[])
    
    # ['tex','mtex','mpose','2dj' ,'lap','CRF_tex','CRF_sil','hm', '3dj','2dj','sil','tex','mshape','mpose', 'open_bone_direc', 'hm_integral', 'kp_cons','tex','mtex','percep']
    #'mpcatex'
    parser.add_argument('--train_datasets', type=list, default=['FreiHand'],#['FreiHand','Obman_hand', 'Obman']
                        help='training dataset to load')#['FreiHand', 'Obman']
    parser.add_argument('--val_datasets', type=list, default=['FreiHand'],
                        help='validation dataset to load')
    parser.add_argument('--is_val', type=bool, default=False)
    parser.add_argument('--val_interval', type=int, default=1)
    

    parser.add_argument('--use_pose_regressor', type=bool, default=False)

    parser.add_argument('--use_discriminator', type=bool, default=False)

    parser.add_argument('--train_batch', type=int, default=8,#32*4,
                        help='training batch size')
    parser.add_argument('--val_batch', type=int, default=8,#16*4,# val-batch = train-batch
                        help='validation batch size')#for testing
    parser.add_argument('--num_workers', type=int, default=8*1,
                        help='num workers')

    parser.add_argument('--pretrain_segmnet', type=str, default=None)  
    #'/code/TextureHand/results/obman/model/seghandnet_29.t7'
    parser.add_argument('--pretrain_model', type=str, default=None)
    #parser.add_argument('--pretrain_model', type=str, default='/code/TextureHand/results/freihand/SSL//2d_3d_integral/model/texturehand_60.t7')
    #parser.add_argument('--pretrain_model', type=str, default='/code/TextureHand/results/freihand/SSL//2d_3d_integral/model/texturehand_40.t7')
    parser.add_argument('--pretrain_texture_model', type=str, default=None)
    #parser.add_argument('--pretrain_texture_model', type=str, default='/code/TextureHand/results/freihand/SSL//2d_3d_texture_perc/model/texturehand_20.t7')

    parser.add_argument('--pretrain_rgb2hm', type=str, default=None)
    #parser.add_argument('--pretrain_rgb2hm', type=str, default='/code/TextureHand/results/freihand/SSL/heatmaps_integral/model/handhm_60.t7')#'/code/TextureHand/saved_models/gcn_hand_model/pretrained_models/net_hm.pth')
    parser.add_argument('--efficientnet_pretrained', type=str, default=None)

    parser.add_argument('--freeze_hm_estimator', type=bool, default=True)#check
    parser.add_argument('--only_train_regressor', type=bool, default=False)
    parser.add_argument('--only_train_texture', type=bool, default=False)#check
    parser.add_argument('--only_train_texture_epochs', type=int, default=0)
    
    

    parser.add_argument('--sigma_val', type=float, default=1e-6)#1e-4
    parser.add_argument('--lambda_laplacian', type=float, default=0.1)#5e-3
    parser.add_argument('--lambda_flatten', type=float, default=5e-4)
    parser.add_argument('--lambda_texture', type=float, default=0.005)#1 0.01
    parser.add_argument('--lambda_silhouette', type=float, default=0.005)#0.1
    parser.add_argument('--lambda_mask', type=float, default=1)#1
    parser.add_argument('--lambda_j2d', type=float, default=1e-3)#0.001
    parser.add_argument('--lambda_j2d_gt', type=float, default=1e-4)#0.001
    parser.add_argument('--lambda_j2d_de', type=float, default=1e-4)#0.001
    parser.add_argument('--lambda_j3d', type=float, default=100)#100
    parser.add_argument('--lambda_j3d_norm', type=float, default=100)
    parser.add_argument('--lambda_verts', type=float, default=100)
    parser.add_argument('--lambda_shape', type=float, default=1e-2)#0.001
    parser.add_argument('--lambda_pose', type=float, default=1e-1)#0.001
    parser.add_argument('--lambda_tex_reg', type=float, default=5e-3)# std: 1e-4
    parser.add_argument('--lambda_pca_text', type=float, default=1e-6)
    parser.add_argument('--lambda_mrgb', type=float, default=1e-3)
    parser.add_argument('--lambda_bone_direc', type=float, default=0.1)
    parser.add_argument('--lambda_percep', type=float, default=1e-5)
    parser.add_argument('--lambda_feature_percep', type=float, default=1e-3)
    parser.add_argument('--lambda_hm', type=float, default=0.001)#1000
    parser.add_argument('--lambda_hm_cons', type=float, default=1)
    parser.add_argument('--lambda_kp_cons', type=float, default=2e-4)
    parser.add_argument('--lambda_depth', type=float, default=1)
    #parser.add_argument('--lambda_laplacian', type=float, default=1)
    parser.add_argument('--lambda_ssim_depth', type=float, default=2*0.1)
    parser.add_argument('--lambda_ssim_tex', type=float, default=0.001)
    parser.add_argument('--lambda_scale', type=float, default=100)
    parser.add_argument('--lambda_trans', type=float, default=100)
    parser.add_argument('--lambda_chamfer', type=float, default=100)
    parser.add_argument('--lambda_depth_pc_chamfer', type=float, default=10)
    parser.add_argument('--lambda_mscale', type=float, default=0.1)

    parser.add_argument('--demo_freq', type=float, default=100)#200
    parser.add_argument('--demo_freq_evaluation', type=float, default=100)#200
    parser.add_argument('--print_freq', type=int, default=100)#500
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--controlled_exp', type=bool, default=False)#Check
    parser.add_argument('--controlled_size', type=int, default=3000)

    parser.add_argument('--total_epochs',type=int,default=100)
    parser.add_argument('--init_lr',type=float,default=0.001)
    parser.add_argument('--lr_steps',type=list,default=[50])
    parser.add_argument('--lr_gamma',type=float,default=0.001)

    parser.add_argument('--frei_selfsup',type=bool,default=False)
    parser.add_argument('--train_queries_frei',type=list,default=[])
    parser.add_argument('--train_queries_rhd',type=list,default=[])
    parser.add_argument('--train_queries_ho3d',type=list,default=[])

    
    parser.add_argument('--semi_ratio',type=float,default=None)
    parser.add_argument('--img_wise_save',type=bool,default=False)
    parser.add_argument('--test_refinement',type=bool,default=False)
    parser.add_argument('--save_2d',type=bool,default=False)
    

    parser.add_argument('--optimizer',type=str,default="Adam")#["Adam","AdamW"]

    # For FreiHand
    parser.add_argument('--freihand_base_path', type=str, default=None, help='Path to where the FreiHAND dataset is located.')
    #image_output = '/code/TextureHand/results/freihand/pic'
    #state_output = '/code/TextureHand/results/freihand/model'
    # For RHD
    parser.add_argument('--rhd_base_path', type=str, default='/data/RHD/RHD_published_v2', help='Path to where the RHD dataset is located.')
    # For HO3D
    parser.add_argument('--ho3d_base_path', type=str, default=None, help='Path to where the RHD dataset is located.')

    # For Obman
    parser.add_argument('--base_path', type=str, default=None, help='Path to where the FreiHAND dataset is located.')
    #parser.add_argument('--out', type=str, default='/code/TextureHand/results/obman/pred.json', help='File to save the predictions.')
    #base_output_dir = '/code/TextureHand/results/freihand/3d'#rgb2hm#'/code/TextureHand/results/obman'
    
    #parser.add_argument('--out', type=str, default=pred_output, help='File to save the predictions.')
    parser.add_argument('--out', type=str, help='File to save the predictions.')

    parser.add_argument('--base_out_path', type=str, default='/home/outputs/freihand/SSL/debug', help='File to save the predictions.')
    parser.add_argument('--image_size', type=int, default=224)#224

    parser.add_argument('--is_write_tb', type=bool, default=False)#224
    parser.add_argument('--writer_topic', type=str, default='/runs/debug/')

    parser.add_argument('--config_json', type=str, default=None)
    
    return parser

def parse():
    parser = get_parser()
    opt = parser.parse_args()
    return opt

def make_output_dir(opt):
    base_output_dir = opt.base_out_path
    opt.base_output_dir = base_output_dir
    opt.image_output = os.path.join(base_output_dir,'pic')
    opt.state_output = os.path.join(base_output_dir,'model')
    opt.obj_output = os.path.join(base_output_dir,'obj')
    opt.pred_output = os.path.join(base_output_dir,'json')
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(opt.image_output, exist_ok=True)
    os.makedirs(opt.state_output, exist_ok=True)
    os.makedirs(opt.obj_output, exist_ok=True)
    os.makedirs(opt.pred_output, exist_ok=True)
    return opt

if __name__ == '__main__':
    opt = parse()
    print('===== arguments: training network =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))