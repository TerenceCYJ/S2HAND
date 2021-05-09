import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

import torch.nn as nn

from src import model
from src import util
#from src.body import Body
from src.hand import Hand

import os
import torch
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class pose_detector(nn.Module):
    def __init__(self):
        super(pose_detector, self).__init__()
        self.hand_estimation = Hand('src/hand_pose_model.pth')
    def forward(self,images):
        #import pdb;pdb.set_trace()
        peaks, values = self.hand_estimation(images)
        return peaks, values

def dump(pred_out_path, all_hand_peaks, all_hand_peaks_values,all_hand_names):
    """ Save predictions into a json file. """
    # make sure its only lists
    xy_pred_list = [x.tolist() for x in all_hand_peaks]
    value_pred_list = [x.tolist() for x in all_hand_peaks_values]
    name_list = [x.tolist() for x in all_hand_names]
    #import pdb; pdb.set_trace()
    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xy_pred_list,
                value_pred_list,
                name_list
            ], fo)
    print('Dumped %d predictions to %s' % (len(all_hand_peaks), pred_out_path))

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


#hand_estimation = Hand('/code/pytorch-openpose/model/hand_pose_model.pth')
# cd .../openpose_detector
# python hand_detect.py
if __name__ == '__main__':
    # for single sample
    '''
    test_image = '/data/FreiHAND_pub_v2/training/rgb/00000000.jpg'
    #test_image = '/code/pytorch-openpose/images/00000014.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    hand_estimation = pose_detector()
    all_hand_peaks = []
    import pdb;pdb.set_trace()
    peaks,values = hand_estimation(images=oriImg)#cyj
    import pdb;pdb.set_trace()
    #peaks = hand_estimation(oriImg)#cyj
    canvas = copy.deepcopy(oriImg)
    all_hand_peaks.append(peaks)
    canvas = util.draw_handpose(canvas, all_hand_peaks)
    import pdb;pdb.set_trace()
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.savefig('/code/pytorch-openpose/images/00000014_out.jpg')
    print('demo out saved!')
    plt.close()
    '''
    

    # for dataset
    image_folder = '/data/FreiHAND_pub_v2/training/rgb'
    save_folder = '/S2HAND/outputs/openpose'
    
    files= os.listdir(image_folder)
    files.sort(key=lambda x: int(x[:8]))
    hand_estimation = pose_detector()
    #import pdb;pdb.set_trace()
    all_hand_peaks = []
    all_hand_peaks_values = []
    all_hand_names = []
    end = time.time()
    for file in files:
        if int(file[:8])>28999 and int(file[:8])<32560:#32560
            #import pdb;pdb.set_trace()
            image_path = os.path.join(image_folder,file)
            oriImg = cv2.imread(image_path)  # B,G,R
            hand_peaks = []
            peaks,values = hand_estimation(images=oriImg)
            #import pdb;pdb.set_trace()
            hand_peaks.append(peaks)
            if int(file[:8])%500 == 0:
                save_img_path = os.path.join(save_folder,'image')
                os.makedirs(save_img_path, exist_ok=True)
                canvas = copy.deepcopy(oriImg)
                canvas = util.draw_handpose(canvas, hand_peaks)
                plt.imshow(canvas[:, :, [2, 1, 0]])
                plt.axis('off')
                plt.savefig(os.path.join(save_img_path,file))
                print('{0} demo out saved!'.format(file))
                plt.close()
                print('Time {0:.3f}\t'.format(time.time() - end))
            all_hand_peaks.append(peaks)
            all_hand_peaks_values.append(values)
            #all_hand_names.append(np.array([int(file[:8])]))
            all_hand_names.append(np.array([file]))
            #import pdb;pdb.set_trace()
    detect_out_path = os.path.join(save_folder,'freihand-train.json')
    dump(detect_out_path,all_hand_peaks,all_hand_peaks_values,all_hand_names)
    