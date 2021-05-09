import torch
import json
import util
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import logging

def load_model(model, args):
    current_epoch = 0
    
    #import pdb; pdb.set_trace()
    if args.pretrain_segmnet is not None:
        state_dict = torch.load(args.pretrain_segmnet)
        model.seghandnet.load_state_dict(state_dict['seghandnet'])
        #current_epoch = 0
        current_epoch = state_dict['epoch']
        print('loading the model from:', args.pretrain_segmnet)
        logging.info('pretrain_segmentation_model: %s' %args.pretrain_segmnet)
    if args.pretrain_model is not None:
        state_dict = torch.load(args.pretrain_model)
        #import pdb; pdb.set_trace()
        # dir(model)
        if 'encoder' in state_dict.keys() and hasattr(model,'encoder'):
            model.encoder.load_state_dict(state_dict['encoder'])
            print('load encoder')
        if 'decoder' in state_dict.keys() and hasattr(model,'hand_decoder'):
            model.hand_decoder.load_state_dict(state_dict['decoder'])
            print('load hand_decoder')
        if 'heatmap_attention' in state_dict.keys() and hasattr(model,'heatmap_attention'):
            model.heatmap_attention.load_state_dict(state_dict['heatmap_attention'])
            print('load heatmap_attention')
        if 'rgb2hm' in state_dict.keys() and hasattr(model,'rgb2hm'):
            model.rgb2hm.load_state_dict(state_dict['rgb2hm'])
            print('load rgb2hm')
        if 'hm2hand' in state_dict.keys() and hasattr(model,'hm2hand'):
            model.hm2hand.load_state_dict(state_dict['hm2hand'])
        if 'mesh2pose' in state_dict.keys() and hasattr(model,'mesh2pose'):
            model.mesh2pose.load_state_dict(state_dict['mesh2pose'])
            print('load mesh2pose')
        
        if 'percep_encoder' in state_dict.keys() and hasattr(model,'percep_encoder'):
            model.percep_encoder.load_state_dict(state_dict['percep_encoder'])
        
        if 'texture_light_from_low' in state_dict.keys() and hasattr(model,'texture_light_from_low'):
            model.texture_light_from_low.load_state_dict(state_dict['texture_light_from_low'])
        if 'textures' in args.train_requires and 'texture_estimator' in state_dict.keys():
            if hasattr(model,'renderer'):
                model.renderer.load_state_dict(state_dict['renderer'])
                print('load renderer')
            if hasattr(model,'texture_estimator'):
                model.texture_estimator.load_state_dict(state_dict['texture_estimator'])
                print('load texture_estimator')
            if hasattr(model,'pca_texture_estimator'):
                model.pca_texture_estimator.load_state_dict(state_dict['pca_texture_estimator'])
                print('load pca_texture_estimator')
        if 'lights' in args.train_requires and 'light_estimator' in state_dict.keys():
            if hasattr(model,'light_estimator'):
                model.light_estimator.load_state_dict(state_dict['light_estimator'])
                print('load light_estimator')
        print('loading the model from:', args.pretrain_model)
        logging.info('pretrain_model: %s' %args.pretrain_model)
        current_epoch = state_dict['epoch']

        if hasattr(model,'texture_light_from_low') and args.pretrain_texture_model is not None:
            texture_state_dict = torch.load(args.pretrain_texture_model)
            model.texture_light_from_low.load_state_dict(texture_state_dict['texture_light_from_low'])
            print('loading the texture module from:', args.pretrain_texture_model)
    # load the pre-trained heat-map estimation model
    if hasattr(model,'rgb2hm') and args.pretrain_rgb2hm is not None:
        #util.load_net_model(args.pretrain_rgb2hm, model.rgb2hm.net_hm)
        #import pdb; pdb.set_trace()
        hm_state_dict = torch.load(args.pretrain_rgb2hm)
        model.rgb2hm.load_state_dict(hm_state_dict['rgb2hm'])
        print('load rgb2hm')
        print('loading the rgb2hm model from:', args.pretrain_rgb2hm)
    #import pdb; pdb.set_trace()
    return model, current_epoch

def freeze_model_modules(model, args):
    if args.freeze_hm_estimator and hasattr(model.module,'rgb2hm'):
        util.rec_freeze(model.module.rgb2hm)
        print("Froze heatmap estimator")
    if args.only_train_regressor:
        if hasattr(model.module,'encoder'):
            util.rec_freeze(model.module.encoder)
            print("Froze encoder")
        if hasattr(model.module,'hand_decoder'):
            util.rec_freeze(model.module.hand_decoder)
            print("Froze hand decoder")
        if hasattr(model.module,'texture_estimator'):
            util.rec_freeze(model.module.texture_estimator)
            print("Froze texture estimator")
    if args.only_train_texture:
        if hasattr(model.module,'rgb2hm'):
            util.rec_freeze(model.module.rgb2hm)
            print("Froze rgb2hm")
        if hasattr(model.module,'encoder'):
            util.rec_freeze(model.module.encoder)
            print("Froze encoder")
        if hasattr(model.module,'hand_decoder'):
            util.rec_freeze(model.module.hand_decoder)
            print("Froze hand decoder")

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













