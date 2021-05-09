from options import train_options
from train_utils import *
import logging
import os
import models_new as models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as torch_f
from data.dataset import get_dataset
from data.datautils import ConcatDataloader
from traineval_util import data_dic, loss_func, save_2d_result,save_2d, mano_fitting, trans_proj, visualize, save_model, write_to_tb, dump
from utils.fh_utils import AverageMeter,EvalUtil


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def TrainVal(mode_train, dat_name, epoch, train_loader, model, optimizer, requires, args, writer=None):
    if mode_train:
        model.train()
        set_name = 'training'
    else:
        model.eval()
        set_name = 'evaluation'

    batch_time = AverageMeter()
    end = time.time()
    # Init output containers
    
    evalutil = EvalUtil()
    xyz_pred_list, verts_pred_list = list(), list()
    op_xyz_pred_list, op_verts_pred_list = list(), list()
    j2d_pred_ED_list,  j2d_proj_ED_list, j2d_detect_ED_list = list(), list(), list() 

    
    for idx, (sample) in enumerate(train_loader):
        # Get batch data
        #import pdb; pdb.set_trace()
        dat_name = sample['dataset']#check
        examples = data_dic(sample, dat_name, set_name, args)
        
        device = examples['imgs'].device
        # Use some algorithm for prediction
        if args.task == 'train' or args.task == 'test' or args.task == 'hm_train':
            outputs = model(images=examples['imgs'], P=examples['Ps'], task=args.task, requires=requires)
        
        # Projection transfer, project to 2D
        outputs, xyz_pred_list, verts_pred_list = trans_proj(outputs, examples['Ks'], dat_name, xyz_pred_list, verts_pred_list)
        #import pdb; pdb.set_trace()
        # Compute loss function
        loss_dic = loss_func(examples,outputs,dat_name,args)
        
        # Compute and backward loss
        loss = torch.zeros(1).float().to(device)
        
        if dat_name == "RHD" and len(args.losses_rhd)>0:
            loss_used = args.losses_rhd
        elif dat_name == "FreiHand" and len(args.losses_frei)>0:
            loss_used = args.losses_frei
        else:
            loss_used = args.losses
        for loss_key in loss_used:
            if loss_dic[loss_key]>0 and (not torch.isnan(loss_dic[loss_key]).sum()):
                loss += loss_dic[loss_key].to(device)
                #print(loss_key,loss_dic[loss_key],loss_dic[loss_key].device)
        
        loss_dic['loss']=loss
        #import pdb; pdb.set_trace()
        if loss < 1e-10:
            continue
        
        if mode_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            op_outputs = None
        else:
            if args.test_refinement:
                op_outputs, op_xyz_pred_list, op_verts_pred_list = mano_fitting(outputs,Ks=examples['Ks'], op_xyz_pred_list=op_xyz_pred_list, op_verts_pred_list=op_verts_pred_list,dat_name=dat_name, args=args)
            else:
                op_outputs = None

        # save 2D results
        if args.save_2d:
            j2d_pred_ED_list, j2d_proj_ED_list, j2d_detect_ED_list = save_2d(examples, outputs, epoch, j2d_pred_ED_list, j2d_proj_ED_list, j2d_detect_ED_list, args)

        # Save visualization and print informations
        batch_time.update(time.time() - end)
        visualize(mode_train,dat_name,epoch,idx,outputs,examples,args, op_outputs = op_outputs, writer=writer, writer_tag=set_name)
        #import pdb; pdb.set_trace()
        # print
        if idx % args.print_freq == 0:
            if optimizer is not None:
                lr_current = optimizer.param_groups[0]['lr']
            else:
                lr_current = 0
            print('Epoch: {0}\t'
                'Iter: [{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                'Loss {loss:.5f}\t'
                'datasat: {dataset:6}\t'
                'lr {lr:.7f}\t'.format(epoch ,idx, len(train_loader),
                                        batch_time=batch_time, loss=loss.data.item(), dataset=dat_name,
                                        lr=lr_current))
            print("Loss backward:\t",['{0}:{1:6f};'.format(loss_item,loss_data.sum()) for loss_item,loss_data in loss_dic.items() if (loss_item in loss_used and loss_data>1e-10)])

            #print("Loss all:\t",['{0}:{1:6f};'.format(loss_item, loss_dic[loss_item].sum().data.item()) for loss_item in loss_dic])
            #print("j3d loss:{0:.4f}; j2d loss:{1:.4f};shape loss:{2:.6f}; pose loss:{3:.6f}; render loss:{4:.6f}; sil loss:{5:.6f}; depth loss:{6:.5f}; render ssim loss:{7:.5f}; depth ssim loss:{8:.5f}; open j2d loss:{9:.5f}; mesh tex std:{10:.10f}; scale loss:{11:.5f}; bone direct loss:{12:.5f}; laplacian loss:{13:.6f}; hm loss:{14:.6f}; kp consistency loss:{15:.6f}; percep loss:{16:.6f}".format(joint_3d_loss.data.item(),joint_2d_loss.data.item(), shape_loss.data.item(),pose_loss.data.item(),texture_loss.data.item(), silhouette_loss.data.item(), depth_loss.data.item(), loss_ssim_tex.data.item(), loss_ssim_depth.data.item(), open_2dj_loss.data.item(), textures_reg.data.item(), mscale_loss.data.item(), open_bone_direc_loss.data.item(),triangle_loss.data.item(),hm_loss.data.item(),kp_cons_loss.data.item(),loss_percep.data.item()))

        # write to tensorboard
        if writer is not None:
            with torch.no_grad():
                write_to_tb(mode_train, writer, loss_dic, epoch, lr=optimizer.param_groups[0]['lr'])
        #break
    # dump results
    if dat_name == 'FreiHand' or dat_name == 'HO3D':
        if mode_train:
            pred_out_path = os.path.join(args.pred_output,'train',str(epoch))
        else:
            pred_out_path = os.path.join(args.pred_output,'test',str(epoch))
            if epoch%args.save_interval==0 and epoch>0:
                os.makedirs(pred_out_path, exist_ok=True)
                pred_out_path_0 = os.path.join(pred_out_path,'pred.json')
                dump(pred_out_path_0, xyz_pred_list, verts_pred_list)
                pred_out_op_path = os.path.join(pred_out_path,'pred_op.json')
                dump(pred_out_op_path, op_xyz_pred_list, op_verts_pred_list)
        if args.save_2d:
            #import pdb;pdb.set_trace()
            save_2d_result(j2d_pred_ED_list, j2d_proj_ED_list, j2d_detect_ED_list, args=args, epoch=epoch)
        #break


def main(base_path, set_name=None, writer = None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = ['evaluation']   

    if 'training' in set_name:#set_name == 'training':
        # initialize train datasets
        train_loaders = []
        if args.controlled_exp:
            # Use subset of datasets so that final dataset size is constant
            limit_size = int(args.controlled_size / len(args.train_datasets))
        else:
            limit_size = None
        for dat_name in args.train_datasets:# iteration = min(dataset_len)/batch_size; go each dataset at a batchsize
            if dat_name == 'FreiHand':
                if len(args.train_queries_frei)>0:
                    train_queries = args.train_queries_frei
                else:
                    train_queries = args.train_queries
                base_path = args.freihand_base_path
            elif dat_name == 'RHD':
                if len(args.train_queries_rhd)>0:
                    train_queries = args.train_queries_rhd
                else:
                    train_queries = args.train_queries
                base_path = args.rhd_base_path
            elif (dat_name == 'Obman') or (dat_name == 'Obman_hand'):
                train_queries = args.train_queries
            elif dat_name == 'HO3D':
                if len(args.train_queries_ho3d)>0:
                    train_queries = args.train_queries_ho3d
                else:
                    train_queries = args.train_queries
                base_path = args.ho3d_base_path
            
            train_dat = get_dataset(
                dat_name,
                'training',#set_name,
                base_path,
                queries = train_queries,
                train = True,
                limit_size=limit_size,
                #transform=transforms.Compose([transforms.Rescale(256),transforms.ToTensor()]))
            )
            print("Training dataset size: {}".format(len(train_dat)))
            # Initialize train dataloader
            
            train_loader0 = torch.utils.data.DataLoader(
                train_dat,
                batch_size=args.train_batch,
                shuffle=True,#check
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            train_loaders.append(train_loader0)
        train_loader = ConcatDataloader(train_loaders)
    #if 'evaluation' in set_name:
    val_loaders = []
    for dat_name_val in args.val_datasets:
        if dat_name_val == 'FreiHand':
            val_queries = args.val_queries
            base_path = args.freihand_base_path
        elif dat_name_val == 'RHD':
            val_queries = args.val_queries
            base_path = args.rhd_base_path
        elif dat_name_val == 'HO3D':
            val_queries = args.val_queries
            base_path = args.ho3d_base_path
        val_dat = get_dataset(
            dat_name_val,
            'evaluation',
            base_path,
            queries = val_queries,
            train = False,
            #transform=transforms.Compose([transforms.Rescale(256),transforms.ToTensor()]))
        )
        print("Validation dataset size: {}".format(len(val_dat)))
        val_loader = torch.utils.data.DataLoader(
            val_dat,
            batch_size=args.val_batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loaders.append(val_loader)
    val_loader = ConcatDataloader(val_loaders)

    #current_epoch = 0
    if len(args.train_datasets) == 1:
        dat_name = args.train_datasets[0]#dat_name
    else:
        dat_name = args.train_datasets

    #losses = AverageMeter()
    if 'training' in set_name:#set_name == 'training':
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(),lr=args.init_lr, betas=(0.9, 0.999), weight_decay=0)
        if args.optimizer == "AdamW":
            optimizer = optim.Adam(model.parameters(),lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

        for epoch in range(1, args.total_epochs+1):
            mode_train = True
            requires = args.train_requires
            args.train_batch = args.train_batch
            TrainVal(mode_train, dat_name, epoch + current_epoch, train_loader, model, optimizer, requires, args, writer)
            torch.cuda.empty_cache()
            
            # save parameters
            if (epoch + current_epoch) % args.save_interval == 0:
                # test
                mode_train = False
                requires = args.test_requires
                args.train_batch = args.val_batch
                print('For test part:')
                TrainVal(mode_train, dat_name_val, epoch + current_epoch, val_loader, model, optimizer, requires, args, writer)
                torch.cuda.empty_cache()
                save_model(model,optimizer,epoch,current_epoch, args)
            scheduler.step()
    elif 'evaluation' in set_name:#set_name == 'evaluation':
        mode_train = False
        requires = args.test_requires
        TrainVal(mode_train, dat_name_val, current_epoch, val_loader, model, None, requires, args, writer)
        print("Finish write prediction. Good luck!")

    print("Done!")
    
if __name__ == '__main__':
    args = train_options.parse()
    
    if args.config_json is not None:
        with open(args.config_json, "r") as f:
            json_dic = json.load(f)
            for parse_key, parse_value in json_dic.items():
                vars(args)[parse_key] = parse_value
    
    args = train_options.make_output_dir(args)
    
    
    #import pdb; pdb.set_trace()#args.update(args.__dic__)

    if args.is_write_tb:
        writer = SummaryWriter(log_dir=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+args.writer_topic+datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        writer = None
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(args.base_output_dir, 'train.log'), level=logging.INFO)
    logging.info("=====================================================")

    model = models.Model('/code/TextureHand/examples/data/obj/sphere/sphere_642.obj', args=args)
    
    model, current_epoch = load_model(model, args)

    model = nn.DataParallel(model.cuda())

    # Optionally freeze parts of the network
    freeze_model_modules(model, args)

    # call with a predictor function
    main(
        args.base_path,
        #args.out,
        set_name=args.mode,
        writer = writer,
    )
    if writer is not None:
        writer.close()