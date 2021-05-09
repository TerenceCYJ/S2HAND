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
- Modify the input and output directory accordingly in ``` ```.
- Offline 2D keypoint detection use a off-the-shelf detector like [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose). 
- - Use the following script  and modify the input and output directory accordingly. 
    ```
    python example/openpose_detector/hand_dectect.py
    ```
- - We also provide detected 2D keypoints for FreiHAND and HO3D training set.
  
    | Dataset | FreiHAND | HO3D |
    | :-----: | :----: | :----: |
    | Detected 2D Keypoints | [GoogleDrive]()/[BaiduNetdisk]() | [GoogleDrive]()/[BaiduNetdisk]() |

### Training
```
cd S2HAND
python ./examples/train.py --config_json examples/config/HO3D/debug.json
```
### Testing

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