# S<sup>2</sup>HAND: Model-based 3D Hand Reconstruction via Self-Supervised Learning

S<sup>2</sup>HAND presents a self-supervised 3D hand reconstruction network that can jointly estimate pose, shape, texture, and the camera viewpoint. Specifically, we obtain geometric cues from the input image through easily accessible 2D detected keypoints. To learn an accurate hand reconstruction model from these noisy geometric cues, we utilize the consistency between 2D and 3D representations and propose a set of novel losses to rationalize outputs of the neural network. For the first time, we demonstrate the feasibility of training an accurate 3D hand reconstruction network without relying on manual annotations. For more details, please see our [paper](https://arxiv.org/abs/2103.11703), [video](https://youtu.be/tuQzu-UfSe8), and [project page](https://terencecyj.github.io/projects/CVPR2021/index.html).

## Code
### Environment
Training is implemented with PyTorch. This code was developed under Python 3.6 and Pytorch 1.1.

Please compile the extension modules by running:
```
pip install tqdm tensorboardX transforms3d chumpy scikit-image

git clone https://github.com/TerenceCYJ/neural_renderer.git
cd neural_renderer
python setup.py install
rm -r neural_renderer
```
Note that we modified the ```neural_renderer/lighting.py``` compared to [daniilidis-group/neural_renderer](https://github.com/daniilidis-group/neural_renderer).

### Data
For example, for 3D hand reconstruction task on the FreiHAND dataset:
- Download the FreiHAND dataset from the [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
- Modify the input and output directory accordingly in ```examples/config/FreiHAND/*.json```.

For HO3D dataset:
- Download the HO3D dataset from the [website](https://www.tugraz.at/index.php?id=40231).
- Modify the input and output directory accordingly in ```examples/config/HO3D/*.json```.

### Offline 2D Detection
- Offline 2D keypoint detection use a off-the-shelf detector like [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose). 
   - We also provide detected 2D keypoints for [FreiHAND training set](https://www.dropbox.com/s/lx9nk8b90a2mgqy/freihand-train.json?dl=0). You may downlad and change the ```self.open_2dj_lists``` in the ```examples/data/dataset.py``` accordingly.
   - Or Download the ```hand_pose_model.pth``` provided by [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose#download-the-models), and put the file to ```examples/openpose_detector/src```. Then use the following script  and modify the input and output directory accordingly. 

        ```python example/openpose_detector/hand_dectect.py```


### Training and Evaluation
#### HO3D
Evaluation: 
download the pretrained model [[texturehand_ho3d.t7]](https://www.dropbox.com/s/q5famyhzu19jv9o/texturehand_ho3d.t7?dl=0), and modify the ```"pretrain_model"``` in ```examples/config/HO3D/evaluation.json```.
```
cd S2HAND
python3 ./examples/train.py --config_json examples/config/HO3D/evaluation.json
```
Training:

Stage-wise training:
```
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-shape.json
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-kp.json
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-finetune.json
```
Or end-to-end training:
```
python3 ./examples/train.py --config_json examples/config/HO3D/SSL-e2e.json
```
Note: remember to check and inplace the dirs and files in the ```*.json``` files.
#### FreiHAND
Evaluation: 
download the pretrained model [[texturehand_freihand.t7]](https://www.dropbox.com/s/kh4xxkfm08bh8py/texturehand_freihand.t7?dl=0), and modify the ```"pretrain_model"``` in ```examples/config/FreiHAND/evaluation.json```.
```
cd S2HAND
python3 ./examples/train.py --config_json examples/config/FreiHAND/evaluation.json
```
Training: refer to HO3D traing scripts.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{chen2021s2hand,
    title={Model-based 3D Hand Reconstruction via Self-Supervised Learning}, 
    author={Chen, Yujin and Tu, Zhigang and Kang, Di and Bao, Linchao and Zhang, Ying and Zhe, Xuefei and Chen, Ruizhi and Yuan, Junsong},
    booktitle={Conference on Computer Vision and Pattern Recognition},
    year={2021}
}
```