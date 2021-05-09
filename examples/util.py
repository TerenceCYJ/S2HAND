#import soft_renderer.functional as srf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from utils.fh_utils import *
import torch.nn as nn
from torch.nn import functional as F

def displaydemo(obj_output, image_output, epoch, idx, vertices, faces, imgs, j2d_gt, j2d, hm_j2d, masks, maskRGBs, render_images,output_joints,joints,re_sil=None, re_img=None, re_depth=None, gt_depth=None,pc_gt_depth=None, pc_re_depth=None, obj_uv6 = None, opt_j2d = None, opt_img=None, dataset_name = 'FreiHand', writer=None, writer_tag='not-sure', img_wise_save=False, refhand=None, warphand=None):
    demo_path = os.path.join(obj_output, 'demo_{:04d}_{:07d}.obj'.format(epoch, idx))
    '''
    if vertices is not None:
        srf.save_obj(demo_path, vertices[0], faces[0])
    '''
    file_str = os.path.join(image_output, '{:04d}_{:07d}.png'.format(epoch, idx))
    fig = plt.figure()
    ax1 = fig.add_subplot(441)
    ax2 = fig.add_subplot(442)
    ax3 = fig.add_subplot(443)
    ax4 = fig.add_subplot(444)
    ax5 = fig.add_subplot(445)
    ax6 = fig.add_subplot(446)
    #ax7 = fig.add_subplot(337, projection='3d')
    #ax8 = fig.add_subplot(338, projection='3d')
    ax7 = fig.add_subplot(447)
    ax8 = fig.add_subplot(448)
    ax11 = fig.add_subplot(4,4,11, projection='3d')
    ax10 = fig.add_subplot(4,4,10, projection='3d')
    #ax11 = fig.add_subplot(4,4,11, projection='3d')
    ax9 = fig.add_subplot(4,4,9)
    ax12 = fig.add_subplot(4,4,12)
    ax13 = fig.add_subplot(4,4,13)
    ax14 = fig.add_subplot(4,4,14)
    ax15 = fig.add_subplot(4,4,15)
    ax16 = fig.add_subplot(4,4,16)
    
    # 11 Image + GT 2D keypints
    ax1.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    ax1.set_title("input image")

    if refhand is not None:
        ax16.imshow(refhand[0].cpu().permute(1, 2, 0).numpy())
        ax16.set_title("ref hand image")
    
    #import pdb; pdb.set_trace()
    if warphand is not None:
        ax15.imshow(warphand[0].cpu().permute(1, 2, 0).numpy())
        ax15.set_title("warp hand image")

    #ax1.axis('off')
    #import pdb; pdb.set_trace()
    if j2d_gt is not None:
        uv = j2d_gt[0].detach().cpu().numpy()
        plot_hand(ax1, uv, order='uv', dataset_name='FreiHand')
    # 12 GT mask
    if masks is not None:
        ax2.imshow(masks[0].cpu().permute(1,2,0).numpy())
    ax2.set_title("GT mask")
    ax2.axis('off')
    # 13 GT maskRGB image
    if maskRGBs is not None:
        ax3.imshow(maskRGBs[0].cpu().permute(1,2,0).numpy())
    ax3.set_title("GT masked image")
    ax3.axis('off')
    # 14 GT Depth
    if gt_depth is not None:
        #import pdb; pdb.set_trace()
        ax4.imshow(gt_depth[0].cpu().detach().numpy())
        #ax6.imshow(gt_depth[0].cpu().detach().permute(1,2,0).numpy())
        #ax9.imshow(re_depth[0].cpu().detach().permute(1,2,0).numpy())
        ax4.set_title("GT Depth")
        ax4.axis('off')
    # 21 Image + output 2D keypints
    ax5.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    if j2d is not None:
        uv_out = j2d[0].detach().cpu().numpy()
        #plot_hand(ax5, uv_out, order='uv', dataset_name=dataset_name)
        plot_hand(ax5, uv_out, order='uv')
        ax5.set_title("output joints in image")
        ax5.axis('off')
    # 22 Rendered mask
    if re_sil is not None:
        re_sil = re_sil[0].repeat(3,1,1)
        ax6.imshow(re_sil.cpu().detach().permute(1,2,0).numpy())
        ax6.set_title("NR Render Sil")
        ax6.axis('off')
    # 23 Rendered RGB image
    if re_img is not None:
        ax7.imshow(re_img[0].cpu().detach().permute(1,2,0).numpy())
        ax7.set_title("NR Render RGB")
        ax7.axis('off')
    # 24 Rendered detph
    if re_depth is not None:
        #import pdb; pdb.set_trace()
        ax8.imshow(re_depth[0].cpu().detach().numpy())
        #ax9.imshow(re_depth[0].cpu().detach().permute(1,2,0).numpy())
        ax8.set_title("NR Render Depth")
        ax8.axis('off')
    # 31 GT 3d Joints
    lims = None
    if joints is not None:
        j3d_gt = joints[0].detach().cpu().numpy()
        #import pdb;pdb.set_trace()
        xlim_min = joints[0,:,0].min()*1.25- joints[0,:,0].max()*0.25
        xlim_max = joints[0,:,0].max()*1.25- joints[0,:,0].min()*0.25
        ylim_min = joints[0,:,1].min()*1.25- joints[0,:,1].max()*0.25
        ylim_max = joints[0,:,1].max()*1.25- joints[0,:,1].min()*0.25
        zlim_min = joints[0,:,2].min()*1.25- joints[0,:,2].max()*0.25
        zlim_max = joints[0,:,2].max()*1.25- joints[0,:,2].min()*0.25
        lims = [xlim_min.cpu().numpy(), xlim_max.cpu().numpy(), ylim_min.cpu().numpy(), ylim_max.cpu().numpy(), zlim_min.cpu().numpy(), zlim_max.cpu().numpy()]
        #ax9=Axes3D(ax9)
        plot_hand_3d(ax11, j3d_gt, order='xyz')
        ax11.set_xlim(lims[0],lims[1])
        ax11.set_ylim(lims[2],lims[3])
        ax11.set_zlim3d(lims[4],lims[5])
        ax11.set_title("GT 3d joints")
    # 32 Output 3d joints
    if output_joints is not None:
        j3d_out = output_joints[0].detach().cpu().numpy()
        #ax7=Axes3D(ax7)
        plot_hand_3d(ax10, j3d_out, order='xyz')
        if lims is not None:
            ax10.set_xlim(lims[0],lims[1])
            ax10.set_ylim(lims[2],lims[3])
            ax10.set_zlim3d(lims[4],lims[5])
        ax10.set_title("Output 3d joints")
    
    # 33 GT one side point cloud
    '''
    if pc_gt_depth is not None or pc_re_depth is not None:
        x = pc_gt_depth[0,:,0].detach().numpy()
        y = pc_gt_depth[0,:,1].detach().numpy()
        z = pc_gt_depth[0,:,2].detach().numpy()
        x1 = pc_re_depth[0,:,0].detach().numpy()
        y1 = pc_re_depth[0,:,1].detach().numpy()
        z1 = pc_re_depth[0,:,2].detach().numpy()
        #ax11 = Axes3D(ax11)
        if pc_gt_depth is not None:
            ax11.scatter(x, y, z, c='g', marker='.', s=0.5,label='GT')
            ax12.scatter(y,-x,c='g', marker='.',s=1, label='GT')
            ax13.scatter(y,z,c='g', marker='.',s=1, label='GT')
            ax14.scatter(x,z,c='g', marker='.',s=1, label='GT')

        if pc_re_depth is not None:
            ax11.scatter(x1, y1, z1, c='r', marker='.', s=0.5,label='re')
            ax12.scatter(y1,-x1,c='r', marker='.', s=1, label='re')
            ax13.scatter(y1,z1,c='r', marker='.', s=1, label='re')
            ax14.scatter(x1,z1,c='r', marker='.', s=1, label='re')
        ax11.set_title("Surface Points")
        ax12.set_title("YX Surface Points")
        ax13.set_title("YZ Surface Points")
        ax14.set_title("XZ Surface Points")
    '''
    # 15 Image + object keypints
    
    ax13.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    #ax13.set_title("object keypoints")
    #ax1.axis('off')
    if obj_uv6 is not None:
        uv = obj_uv6[0].detach().cpu().numpy()
        ax13.scatter(uv[:,0],uv[:,1])

    # 21 Image + output 2D keypints
    ax9.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    if hm_j2d is not None:
        uv_out = hm_j2d[0].detach().cpu().numpy()
        #plot_hand(ax5, uv_out, order='uv', dataset_name=dataset_name)
        plot_hand(ax9, uv_out, order='uv')
        ax9.set_title("heatmap keypoints")
        ax9.axis('off')


    # 14 for opti 2dj
    ax14.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    if opt_j2d is not None:
        uv = opt_j2d[0].detach().cpu().numpy()
        plot_hand(ax14, uv, order='uv', dataset_name=dataset_name)
        ax14.set_title("optimized projected")
        ax14.axis('off')
    if opt_img is not None:
        ax15.imshow(opt_img[0].cpu().permute(1, 2, 0).numpy())
        ax15.axis('off')
    # 34 output one side point cloud
    # segmentations
    '''
    if output_segms is not None:
        #show_segm = torch.cat((output_segms[0].cpu(), torch.zeros(1,output_segms.shape[2],output_segms.shape[3])),dim=0)
        #import pdb; pdb.set_trace()
        _, pred = torch.max(output_segms[0],dim=0)
        #import pdb; pdb.set_trace()
        pred = torch.cat((pred.unsqueeze(0).cpu(), torch.zeros([2,pred.shape[0],pred.shape[1]],dtype=pred.dtype)),dim=0)
        ax4.imshow(pred.cpu().detach().permute(1,2,0).numpy())
        #ax4.imshow(output_segms[0,0].repeat(3,1,1).cpu().detach().permute(1,2,0).numpy())
        ax4.set_title("output segms")
    
    #ax5.imshow(torch.squeeze(render_images[0],0)[0:3].cpu().detach().permute(1,2,0).numpy())
    #ax5.set_title("output image")
    ax5.imshow(segms_gt[0].unsqueeze(0).repeat(3,1,1).cpu().detach().permute(1,2,0).numpy())
    ax5.set_title("GT segms")
    '''
    #import pdb; pdb.set_trace()
    
    
    '''
    if render_images is not None:
        ax5.imshow(render_images[0,0:3].cpu().detach().permute(1,2,0).numpy())
        ax5.set_title("Render image")
        sil = render_images[0,3].repeat(3,1,1)
        ax6.imshow(sil.cpu().detach().permute(1,2,0).numpy())
        #ax6.imshow(render_silhouettes[0,0:3].cpu().detach().permute(1,2,0).numpy())
        ax6.set_title("Render silhouettes")
    '''
    #import pdb; pdb.set_trace()
    
    '''
    lims = None
    if joints is not None:
        j3d_gt = joints[0].detach().cpu().numpy()
        #import pdb;pdb.set_trace()
        xlim_min = joints[0,:,0].min()*1.25- joints[0,:,0].max()*0.25
        xlim_max = joints[0,:,0].max()*1.25- joints[0,:,0].min()*0.25
        ylim_min = joints[0,:,1].min()*1.25- joints[0,:,1].max()*0.25
        ylim_max = joints[0,:,1].max()*1.25- joints[0,:,1].min()*0.25
        zlim_min = joints[0,:,2].min()*1.25- joints[0,:,2].max()*0.25
        zlim_max = joints[0,:,2].max()*1.25- joints[0,:,2].min()*0.25
        lims = [xlim_min.cpu().numpy(), xlim_max.cpu().numpy(), ylim_min.cpu().numpy(), ylim_max.cpu().numpy(), zlim_min.cpu().numpy(), zlim_max.cpu().numpy()]
        #ax7=Axes3D(ax7)
        plot_hand_3d(ax7, j3d_gt, order='xyz')
        ax7.set_xlim(lims[0],lims[1])
        ax7.set_ylim(lims[2],lims[3])
        ax7.set_zlim(lims[4],lims[5])
        ax7.set_title("GT 3d joints")
    if output_joints is not None:
        j3d_out = output_joints[0].detach().cpu().numpy()
        #ax7=Axes3D(ax7)
        plot_hand_3d(ax8, j3d_out, order='xyz')
        if lims is not None:
            ax8.set_xlim(lims[0],lims[1])
            ax8.set_ylim(lims[2],lims[3])
            ax8.set_zlim(lims[4],lims[5])
        ax8.set_title("Output 3d joints")
    '''
    
    
    
    
    
    '''
    re_silhouettes = render_silhouettes[0,3]
    re_silhouettes = torch.cat((re_silhouettes.unsqueeze(0).cpu(), torch.zeros([2,re_silhouettes.shape[0],re_silhouettes.shape[1]],dtype=re_silhouettes.dtype)),dim=0)
    ax8.imshow(re_silhouettes.detach().permute(1,2,0).numpy())
    #'''
    
    file_str = os.path.join(image_output, '{:04d}_{:07d}.png'.format(epoch, idx))
    #plt.tight_layout()
    plt.savefig(file_str,dpi=800)

    if writer is not None:
        writer.add_figure(writer_tag, fig)

    plt.close()
    print("save image at:", file_str)
    
    '''
    if 'segms' in args.train_requires:
        # save raw images
        save_rgb_path = os.path.join(image_output, '{:03d}_{:07d}_rgb.png'.format(epoch, idx))
        image = imgs[0].cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        image.save(save_rgb_path)

        # save segm images
        save_segm_gt_path = os.path.join(image_output, '{:03d}_{:07d}_segm_gt.png'.format(epoch, idx))
        segms_gt = segms_gt.type(torch.FloatTensor)
        image = segms_gt[0].cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        image.save(save_segm_gt_path)

        save_segm_pre_path = os.path.join(image_output, '{:03d}_{:07d}_segm_pre.png'.format(epoch, idx))
        #image = output_segms[0].cpu().clone()
        _, pred = torch.max(output_segms[0].cpu(),dim=0)
        #import pdb; pdb.set_trace()
        pred = torch.cat((pred.unsqueeze(0).cpu(), torch.zeros([2,pred.shape[0],pred.shape[1]],dtype=pred.dtype)),dim=0)
        #
        #image = pred.squeeze(0)
        #image = unloader(image)
        image = unloader(pred.type(torch.FloatTensor))
        image.save(save_segm_pre_path)

        print("save image at:", image_output)
    '''
    #import pdb; pdb.set_trace()
    if img_wise_save:
        file_path = os.path.join(image_output, "per_images")
        os.makedirs(file_path, exist_ok=True)
        # raw figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        file_str = os.path.join(file_path, '{:04d}_{:07d}_raw_img.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)

        # detected keypoints
        if j2d_gt is not None:
            uv = j2d_gt[0].detach().cpu().numpy()
            plot_hand(ax, uv, order='uv', dataset_name='FreiHand',linewidth=3,markersize=10) 
        file_str = os.path.join(file_path, '{:04d}_{:07d}_detect_kp.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        #import pdb; pdb.set_trace()
        # only detected keypoints
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(torch.ones_like(imgs[0]).cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        if j2d_gt is not None:
            uv = j2d_gt[0].detach().cpu().numpy()
            plot_hand(ax, uv, order='uv', dataset_name='FreiHand',linewidth=3,markersize=10) 
        file_str = os.path.join(file_path, '{:04d}_{:07d}_detect_kp_nbg.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()
        
        # output keypoints
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        if j2d is not None:
            uv_out = j2d[0].detach().cpu().numpy()
            plot_hand(ax, uv_out, order='uv', dataset_name='FreiHand',linewidth=3,markersize=10)
        file_str = os.path.join(file_path, '{:04d}_{:07d}_proj_kp.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        # only output keypoints
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(torch.ones_like(imgs[0]).cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        if j2d is not None:
            uv = j2d[0].detach().cpu().numpy()
            plot_hand(ax, uv, order='uv', dataset_name='FreiHand',linewidth=3,markersize=10) 
        file_str = os.path.join(file_path, '{:04d}_{:07d}_proj_kp_nbg.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        # rendered image
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        #import pdb; pdb.set_trace()
        if re_img is not None:
            #re_img_this = torch.where(re_img<1e-10,torch.ones_like(re_img).to(re_img.device).float(),re_img)
            ax.imshow(re_img[0].cpu().detach().permute(1,2,0).numpy())
        ax.axis('off')
        file_str = os.path.join(file_path, '{:04d}_{:07d}_re_img.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        # optimize 2dj image
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        if opt_j2d is not None:
            uv_out = opt_j2d[0].detach().cpu().numpy()
            plot_hand(ax, uv_out, order='uv', dataset_name='FreiHand',linewidth=3,markersize=10)
        file_str = os.path.join(file_path, '{:04d}_{:07d}_proj_kp_op.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        # only optimize keypoints
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(torch.ones_like(imgs[0]).cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
        if opt_j2d is not None:
            uv = opt_j2d[0].detach().cpu().numpy()
            plot_hand(ax, uv, order='uv', dataset_name='FreiHand',linewidth=3,markersize=10) 
        file_str = os.path.join(file_path, '{:04d}_{:07d}_proj_kp_op_nbg.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()


        # optimize rendered image
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        if opt_img is not None:
            ax.imshow(opt_img[0].cpu().detach().permute(1,2,0).numpy())
        ax.axis('off')
        file_str = os.path.join(file_path, '{:04d}_{:07d}_re_img_op.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        # rendered sil
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        if re_sil is not None:
            re_sil = re_sil[0].repeat(3,1,1)
            ax.imshow(re_sil.cpu().detach().permute(1,2,0).numpy())
        ax.axis('off')
        file_str = os.path.join(file_path, '{:04d}_{:07d}_re_sil.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()


        # rendered sil * RGB
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        if maskRGBs is not None:
            ax.imshow(maskRGBs[0].cpu().detach().permute(1,2,0).numpy())
        ax.axis('off')
        file_str = os.path.join(file_path, '{:04d}_{:07d}_raw_fg.png'.format(epoch, idx))
        plt.savefig(file_str,dpi=300)
        plt.close()

        # output keypoints in other viewpoint

        # rendered image in other viewpoint
        print("save per image at:", file_path)
        #import pdb; pdb.set_trace()


def displadic(obj_output, image_output, epoch, idx, examples, outputs, dat_name, op_outputs=None, opt_j2d=None, opt_img=None, writer=None, writer_tag='not-sure', img_wise_save=False):
    vertices = outputs['vertices'] if 'vertices' in outputs else None
    faces = outputs['faces'] if 'faces' in outputs else None
    imgs = examples['imgs'] if 'imgs' in examples else None
    open_2dj = examples['open_2dj'] if 'open_2dj' in examples else None
    j2d_gt = examples['j2d_gt'] if 'j2d_gt' in examples else None
    j2d = outputs['j2d'] if 'j2d' in outputs else None
    hm_j2d_list = outputs['hm_j2d_list'] if 'hm_j2d_list' in outputs else [None]
    masks = examples['masks'] if 'masks' in examples else None
    maskRGBs = outputs['maskRGBs'] if 'maskRGBs' in outputs else None
    output_joints = outputs['output_joints'] if 'output_joints' in outputs else None
    joints = outputs['joints'] if 'joints' in outputs else None
    re_sil = outputs['re_sil'] if 're_sil' in outputs else None
    re_img = outputs['re_img'] if 're_img' in outputs else None
    re_depth = outputs['re_depth'] if 're_depth' in outputs else None
    gt_depth = examples['gt_depth'] if 'gt_depth' in examples else None
    pc_gt_depth = examples['pc_gt_depth'] if 'pc_gt_depth' in examples else None
    pc_re_depth = examples['pc_gt_depth'] if 'pc_gt_depth' in examples else None
    refhand = examples['refhand'] if 'refhand' in examples else None
    warphand = outputs['warphand'] if 'warphand' in outputs else None
    
    #hm_j2d_list = outputs['hm_j2d_list'] if 'hm_j2d_list' in outputs else None
    #import pdb; pdb.set_trace()
    if op_outputs is not None:
        #opt_img = 
        opt_j2d = op_outputs['j2d']
        opt_img = op_outputs['re_img']

    if open_2dj is not None:
        displaydemo(obj_output, image_output, epoch, idx, vertices, faces, imgs, open_2dj, j2d, hm_j2d_list[-1], masks, maskRGBs, None, output_joints,joints,re_sil, re_img, re_depth,gt_depth,pc_gt_depth, pc_re_depth, opt_j2d = opt_j2d, opt_img=opt_img, dataset_name=dat_name, writer=writer, writer_tag=writer_tag, img_wise_save=img_wise_save, refhand=refhand, warphand = warphand)
    else:
        displaydemo(obj_output, image_output, epoch, idx, vertices, faces, imgs, j2d_gt, j2d, hm_j2d_list[-1], masks, maskRGBs, None, output_joints,joints,re_sil, re_img, re_depth,gt_depth,pc_gt_depth, pc_re_depth, opt_j2d = opt_j2d, opt_img=opt_img, dataset_name=dat_name, writer=writer, writer_tag=writer_tag, img_wise_save=img_wise_save, refhand=refhand, warphand = warphand)
    
    '''
    if op_outputs is not None:
        image_output_op = os.path.join(image_output,"op")
        displaydemo(obj_output, image_output_op, epoch, idx, op_outputs['vertices'], faces, imgs, open_2dj, op_outputs['j2d'], hm_j2d_list[-1], masks, maskRGBs, None, output_joints,joints,re_sil, re_img, re_depth,gt_depth,pc_gt_depth, pc_re_depth, opt_j2d = opt_j2d, opt_img=opt_img, dataset_name=dat_name, writer=writer, writer_tag=writer_tag, img_wise_save=img_wise_save)
    '''

def multiview_render(image_output, outputs, epoch, idx):
    #import pdb; pdb.set_trace()
    batch_size = outputs['vertices'].shape[0]
    device = outputs['vertices'].device
    from utils.hand_3d_model import rodrigues
    from torchvision.utils import save_image
    file_path = os.path.join(image_output, "multiviews")
    os.makedirs(file_path, exist_ok=True)
    
    for i in range(20):
        rots = torch.tensor([0,np.pi*2/20*i,0]).unsqueeze(0).repeat(batch_size,1).to(device)
        Rots = rodrigues(rots)[0]
        new_vertices = torch.matmul(Rots,(outputs['vertices']-torch.mul(outputs['joints'][:,9],torch.tensor([0,0,1]).float().to(device)).unsqueeze(1)).permute(0,2,1)).permute(0,2,1) + outputs['trans'].unsqueeze(1)
        #new_vertices = torch.matmul(Rots,(outputs['vertices']-outputs['joints'][:,9].unsqueeze(1)).permute(0,2,1)).permute(0,2,1) + outputs['trans'].unsqueeze(1)
        #new_vertices = torch.matmul(Rots,(outputs['vertices']-outputs['trans'].unsqueeze(1)).permute(0,2,1)).permute(0,2,1) + outputs['trans'].unsqueeze(1)
        op_re_img,op_re_depth,op_re_sil = outputs['render'](new_vertices, outputs['faces'], torch.tanh(outputs['face_textures']), mode=None)
        #op_re_img,op_re_depth,op_re_sil = outputs['render'](outputs['vertices'], outputs['faces'], torch.tanh(outputs['face_textures']), mode=None)
        file_str = os.path.join(file_path, '{:04d}_{:07d}_h_{}.png'.format(epoch, idx,i))
        save_image(op_re_img[0], file_str)
        rots = torch.tensor([np.pi*2/20*i,0,0]).unsqueeze(0).repeat(batch_size,1).to(device)
        Rots = rodrigues(rots)[0]
        new_vertices = torch.matmul(Rots,(outputs['vertices']-torch.mul(outputs['joints'][:,9],torch.tensor([0,0,1]).float().to(device)).unsqueeze(1)).permute(0,2,1)).permute(0,2,1) + outputs['trans'].unsqueeze(1)
        op_re_img,op_re_depth,op_re_sil = outputs['render'](new_vertices, outputs['faces'], torch.tanh(outputs['face_textures']), mode=None)
        file_str = os.path.join(file_path, '{:04d}_{:07d}_v_{}.png'.format(epoch, idx,i))
        save_image(op_re_img[0], file_str)
    
    

    '''
    for i in range(20):
        new_render = outputs['render']
        #new_render.light_intensity_ambient /= 100
        new_render.light_direction[:,0] = np.pi*2/20*i
        op_re_img,op_re_depth,op_re_sil = new_render(outputs['vertices'], outputs['faces'], torch.tanh(outputs['face_textures']), mode=None)
        file_str = os.path.join(file_path, '{:04d}_{:07d}_light_{}.png'.format(epoch, idx,i))
        save_image(op_re_img[0], file_str)
    '''

    return 0


def display_img_2dj(image_output,file_name,imgs,j2d_gt, dataset_name=None):
    file_str = os.path.join(image_output, file_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    #ax1.imshow(imgs[0].cpu().numpy())
    if j2d_gt is not None:
        uv = j2d_gt[0].detach().cpu().numpy()
        plot_hand(ax1, uv, order='uv', dataset_name=dataset_name)
    plt.savefig(file_str)
    plt.close()
    print("save image at:", file_str)


def displayRHD(obj_output, image_output, epoch, idx,imgs,j2d_l=None,j2d_r=None,mask_l=None,mask_r=None,depth_l=None,depth_r=None,sides_l=None,sides_r=None):
    demo_path = os.path.join(obj_output, 'demo_{:03d}_{:07d}.obj'.format(epoch, idx))
    file_str = os.path.join(image_output, '{:03d}_{:07d}.png'.format(epoch, idx))
    fig = plt.figure()
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    ax1.imshow(imgs[0].cpu().permute(1, 2, 0).numpy())
    ax1.set_title("input image")
    if j2d_l is not None and sides_l[0].data.item()>0:
        uv = j2d_l[0].detach().cpu().numpy()
        plot_hand(ax1, uv, order='uv')
    if j2d_r is not None and sides_r[0].data.item()>0:
        uv = j2d_r[0].detach().cpu().numpy()
        plot_hand(ax1, uv, order='uv')
    if mask_l is not None:
        ax2.imshow(mask_l[0].cpu().numpy())
    ax2.set_title("mask left")
    ax2.axis('off')
    if mask_r is not None:
        ax3.imshow(mask_r[0].cpu().numpy())
    ax3.set_title("mask right")
    ax3.axis('off')
    if depth_l is not None:
        ax4.imshow(depth_l[0].cpu().numpy())
    ax4.set_title("mask left")
    ax4.axis('off')
    if depth_r is not None:
        ax5.imshow(depth_r[0].cpu().numpy())
    ax5.set_title("mask right")
    ax5.axis('off')
    
    file_str = os.path.join(image_output, '{:03d}_{:07d}.png'.format(epoch, idx))
    plt.savefig(file_str)
    plt.close()
    print("save image at:", file_str)


def load_net_model(model_path, net):
    assert (os.path.isfile(model_path)), ('The model does not exist! Error path:\n%s' % model_path)

    model_dict = torch.load(model_path, map_location='cpu')
    module_prefix = 'module.'
    module_prefix_len = len(module_prefix)

    for k in model_dict.keys():
        if k[:module_prefix_len] != module_prefix:
            net.load_state_dict(model_dict)
            return 0

    del_keys = filter(lambda x: 'num_batches_tracked' in x, model_dict.keys())
    for k in del_keys:
        del model_dict[k]

    model_dict = OrderedDict([(k[module_prefix_len:], v) for k, v in model_dict.items()])
    net.load_state_dict(model_dict)
    return 0

def find_keypoints_max(heatmaps):
    """
    heatmaps: C x H x W
    return: C x 3
    """
    # flatten the last axis
    heatmaps_flat = heatmaps.view(heatmaps.size(0), -1)

    # max loc
    max_val, max_ind = heatmaps_flat.max(1)
    max_ind = max_ind.float()

    max_v = torch.floor(torch.div(max_ind, heatmaps.size(1)))
    max_u = torch.fmod(max_ind, heatmaps.size(2))
    return torch.cat((max_u.view(-1,1), max_v.view(-1,1), max_val.view(-1,1)), 1)

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))#[1,B*21,1,height,width]
    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)#[1,B*21,width=256]
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)#[1,B*21,hight=256]
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)#[1,B*21,depth=1]
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    #import pdb; pdb.set_trace()
    return accu_x, accu_y, accu_z

def softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth):
    # global soft max
    #preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = preds.reshape((1, num_joints, -1))
    preds = F.softmax(preds, 2)
    # integrate heatmap into joint location
    x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    #x = x / float(hm_width) - 0.5
    #y = y / float(hm_height) - 0.5
    #z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

def compute_uv_from_integral(hm, resize_dim):
    """
    https://github.com/JimmySuen/integral-human-pose
    
    :param hm: B x K x H x W (Variable)
    :param resize_dim:
    :return: uv in resize_dim (Variable)
    
    heatmaps: C x H x W
    return: C x 3
    """
    upsample = nn.Upsample(size=resize_dim, mode='bilinear', align_corners=True)  # (B x K) x H x W
    resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1])
    #import pdb; pdb.set_trace()
    num_joints = resized_hm.shape[0]
    hm_width = resized_hm.shape[-1]
    hm_height = resized_hm.shape[-2]
    hm_depth = 1
    pred_jts = softmax_integral_tensor(resized_hm, num_joints, hm_width, hm_height, hm_depth)
    pred_jts = pred_jts.view(-1,hm.size(1), 3)
    #import pdb; pdb.set_trace()
    return pred_jts
    

def compute_uv_from_heatmaps(hm, resize_dim):
    """
    :param hm: B x K x H x W (Variable)
    :param resize_dim:
    :return: uv in resize_dim (Variable)
    """
    upsample = nn.Upsample(size=resize_dim, mode='bilinear', align_corners=True)  # (B x K) x H x W
    resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1])

    uv_confidence = find_keypoints_max(resized_hm)  # (B x K) x 3
    return uv_confidence.view(-1, hm.size(1), 3)

def compute_heatmaps_from_uv(uv, resize_dim):
    """
    :param uv: B x 21 x 2
    :param resize_dim: [64,64]
    :return: hm: B x K x H x W (Variable)
    """
    
    batchsize = uv.shape[0]
    num_parts = uv.shape[1]
    output_res = 256
    sigma = output_res/64
    size = 6*sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3*sigma + 1, 3*sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    hms = np.zeros(shape = (batchsize, num_parts, output_res, output_res), dtype = np.float32)
    #import pdb; pdb.set_trace()
    for i in range(batchsize):
        for p in range(num_parts):
            x, y = int(uv[i,p,0]), int(uv[i,p,1])
            if x<0 or y<0 or x>=output_res or y>=output_res:
                continue
            ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
            br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)
            c,d = max(0, -ul[0]), min(br[0], output_res) - ul[0]
            a,b = max(0, -ul[1]), min(br[1], output_res) - ul[1]

            cc,dd = max(0, ul[0]), min(br[0], output_res)
            aa,bb = max(0, ul[1]), min(br[1], output_res)

            hms[i, p, aa:bb,cc:dd] = np.maximum(hms[i, p, aa:bb,cc:dd], g[a:b,c:d])
    hms = torch.from_numpy(hms).to(uv.device)
    upsample = nn.Upsample(size=resize_dim, mode='bilinear', align_corners=True)
    hms = upsample(hms)
    #import pdb; pdb.set_trace()
    return hms#[b,21,64,64]

def rec_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d