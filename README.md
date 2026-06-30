<br />
<p align="center">
  <h1 align="center">Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Guided by an Event Camera
(ICRA'24)</h1>
  <p align="center" >
    Jiahang Cao,
    Xu Zheng,
    Yuanhuiyi Lyu,
    Jiaxu Wang,
    Renjing Xu<sup>†</sup>,
    Lin Wang<sup>†</sup>
  </p>
  <p align="center" >
    <em>HKUST(GZ) & HKUST</em> 
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2309.09297'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2309.09297' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Proceeding-HTML-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Proceeding Supp'>
    </a>
  </p>
  <p align="center">
    <img src="figs/main.png" alt="Logo" width="99%">
  </p>
</p>



## Requirements

1. (Optional) Creating conda environment.
```shell
conda create -n EOLO python=3.8 -y
conda activate EOLO
```

2. Installing dependencies.
```shell
git clone https://github.com/SageCao1125/EOLO.git
cd EOLO
pip install -r requirements.txt
```

Some recent Python environments may also need the following packages:
```shell
pip install spikingjelly mmcv-lite mmengine timm pycocotools thop wandb
```

## Training
**[Update 24.08.05]** The checkpoint of EOLO in under-exposure scene in VOC is now released. You can download the checkpoint through [this link](https://drive.google.com/drive/folders/1Q9L7dzf82zRGWOyka3hnJqhTjKFn0Vog?usp=drive_link).

**[Update 26.06.30]** If you use the current cleaned code, please use `EOLO_50_60.61_corrected_26.06.30.pth` [this link](https://drive.google.com/file/d/1WMncJ0AGa6R1swRzPI0kW3gh7DYwDoFO/view?usp=drive_link) for inference. This file has the same learned weights as the original released `EOLO_50_60.61.pth`, but its state dict keys have been updated to match the current module names. See [Checkpoint note for inference](#checkpoint-note-for-inference).

We also re-ran the full under-exposure VOC reproduction from scratch and confirmed that the code, data generation pipeline, and training settings are working as expected. The reproduced result is very close to the reported checkpoint result. Please see [Reproduction Check, June 2026](#reproduction-check-june-2026) for the command, environment, log, and numbers.

Codes for training EOLO: 

```shell
CUDA_VISIBLE_DEVICES=0 python train_eyolo.py \
     -d voc \
     --cuda \
     -m E-yolo-tiny \
     --ema \
     --num_gpu 1 \
     --batch_size 32 \
     --root path/to/dataset/\
     --lr 0.0005 \
     --img_size 320 \
     --max_epoch 50 \
     --lr_epoch 30 40 \
     --save_name EOLO-tiny_VOC_Underexposure_0.2_random42_1gpu_32bs_50epoch_SREF\
     --img_size 320\
     --data_type Exposure_Event\
     --exposure_factor Underexposure_0.2_random42\
     --fusion_method SREF\
     --use_wandb   
```

## Visualization

**[Update 25.03.15]** Codes for test EOLO's detection results:

```shell
python test_eyolo.py -d voc \
        --cuda \
        -m E-yolo-tiny  \
        --weight EOLO_50_60.61_current_code.pth \
        --img_size 320 \
        --root path/to/dataset/ \
        --save_name EOLO_results\
        --data_type Exposure_Event\
        --fusion_method SREF\
        --exposure_factor Underexposure_0.2_random42 \
        --visual_threshold 0.4  
```
Feel free to upload any images (RGB-E) you wish to be detected.

## Dataset Preparation
### Download VOC 2007 & 2012 dataset
```shell
# Please specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh
sh data/scripts/VOC2012.sh
```


## Event-based Dataset Generation
<p align="center">
  <img src="figs/event2frame.png" alt="Logo" width="50%">
</p>

To obtain paired event data, we propose a novel event frame synthesis method that generates event frames by the randomized optical flow and luminance gradients. **Only a single RGB/HDR image is required to generate the corresponding event frames.**

You can easily generate E-VOC dataset by 

```shell
python event2frame.py
```
The resulting dataset will have the following data structure:

``` graphql
VOC2007
|---Event                      ## Raw Event (.npy)
   |---{event_type}, e.g.,'Underexposure_0.2_random42'
       |---XXXX.npy
       |...
|---EventFrameImages           ## Event Frame (.jpg)
    |---{event_type}
       |---XXXX.jpg
       |...
|---ExposureImages             ## Exposure RGB image for visulization (.jpg), clip into [0,255] from HDR image
    |---{event_type}
       |---XXXX.jpg
       |...
|---HDRImages                  ## Exposure Images (.exr)
    |---{event_type}
       |---XXXX.exr
       |...
|---Annotations                
|---JPEGImages
|---ImageSets
|---SegmentationClass
|---SegmentationObject
```
where the Event, EventFrameImages, ExposureImages and HDRImages are newly generated. Please remember, you need to first download the original VOC dataset before this step. 

## Reproduction Check, June 2026

We re-ran the under-exposure VOC setting from scratch to make sure the released result can be reproduced.

The full training log is included at [`log_files/eolo_repro_2026.log`](log_files/eolo_repro_2026.log). It contains the complete command output, the successful pretrained backbone loading message, the epoch 50 evaluation, and the saved checkpoint line.

### Data

VOC2007 trainval/test and VOC2012 trainval were downloaded with the scripts in `data/scripts/`. Event data was generated with `event2frame.py` using:

```text
EXPOSE = 0.2
random_optical_flow = True
np.random.seed(42)
event type = Underexposure_0.2_random42
```

Please generate the event data for both `VOC2007` and `VOC2012`, because the default training command uses both:

```text
VOC2007 trainval + VOC2012 trainval
```

After generation, the checked file counts were:

```text
VOC2007 JPEGImages=9963 Event=9963 EventFrameImages=9963 ExposureImages=9963 HDRImages=9963
VOC2012 JPEGImages=17125 Event=17125 EventFrameImages=17125 ExposureImages=17125 HDRImages=17125
```

### Backbone initialization

EOLO uses the CSPDarkNet-Tiny backbone from [PyTorch_YOLO-Family](https://github.com/yjh0410/PyTorch_YOLO-Family). Download the pretrained backbone before training:

```shell
mkdir -p weights/cspdarknet_tiny
wget -O weights/cspdarknet_tiny/cspdarknet_tiny.pth \
  https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/cspdarknet_tiny.pth
```

A correct training log should contain:

```text
The pretrained weight of cspdarknet_tiny is found successfully ...
```

If you do not see this line, the RGB backbone is being trained from scratch, which is not the setting used for the reproduced result below.

### Training command

The reproduction run used the following command:

```shell
CUDA_VISIBLE_DEVICES=0 python train_eyolo.py \
     -d voc \
     --cuda \
     -m E-yolo-tiny \
     --ema \
     --num_gpu 1 \
     --batch_size 32 \
     --root path/to/VOCdevkit \
     --lr 0.0005 \
     --img_size 320 \
     --max_epoch 50 \
     --lr_epoch 30 40 \
     --save_name EOLO_repro_Underexposure_0.2_random42_1gpu_32bs_50epoch_SREF \
     --data_type Exposure_Event \
     --exposure_factor Underexposure_0.2_random42 \
     --fusion_method SREF
```

The run evaluates on VOC2007 test at epoch 50.

### Reference environment

The reproduction was checked on one NVIDIA RTX 4090. The exact local environment was:

```text
Python 3.13.9
PyTorch 2.7.1+cu126
Torchvision 0.22.1+cu126
OpenCV 4.12.0
NumPy 2.2.5
spikingjelly 0.0.0.0.14
mmcv-lite 2.2.0
mmengine 0.10.7
timm 1.0.27
pycocotools 2.0.11
thop 0.1.1
```

The original code was developed with an older Python stack, but the numbers below were reproduced with the environment above.

### Results

The released corrected checkpoint gives:

```text
EOLO_50_60.61_current_code.pth
VOC2007 test AP50 = 60.60
VOC2007 test AP75 = 32.57
VOC2007 test AP95 = 1.07
```

The fresh reproduction training run gives:

```text
E-yolo-tiny_50_60.08.pth
VOC2007 test AP50 = 60.08
VOC2007 test AP75 = 31.98
VOC2007 test AP95 = 1.26
```

The AP50 difference is about 0.5 points, which is within the expected variation for this training setup. The main takeaway is that the released result is reproducible when the event data, CSPDarkNet-Tiny pretrained backbone, and checkpoint key names are all handled correctly.

The reproduced checkpoint from this run was saved as:

```text
weights/voc/E-yolo-tiny/EOLO_repro_2026_Underexposure_0.2_random42_1gpu_32bs_50epoch_SREF_pretrained/E-yolo-tiny_50_60.08.pth
```

## Checkpoint Note For Inference

Please read this part if your inference result looks much worse than expected.

There was a small but important naming mismatch between the first released checkpoint and the cleaned module names in this repository.

The original released checkpoint `EOLO_50_60.61.pth` stores the fusion module weights with old key names such as:

```text
fusion_s.AFNet.*
fusion_m.AFNet.*
fusion_l.AFNet.*
```

In the current code, the same fusion block is registered under:

```text
fusion_s.SREF.*
fusion_m.SREF.*
fusion_l.SREF.*
```

The model architecture is the same. The problem is only the key name in the checkpoint. If the old checkpoint is loaded with `strict=False`, PyTorch will not stop the run, but many fusion weights will be skipped silently. In that case the model can run, but inference quality can look much worse than expected.

In our check, loading the old checkpoint directly with the cleaned code produced many skipped fusion weights. After remapping the keys, the same released weights gave `60.60` AP50 on VOC2007 test, matching the checkpoint name.

To avoid this, please use:

```text
EOLO_50_60.61_corrected_26.06.30.pth
```

This checkpoint was produced by remapping `.AFNet.` to `.SREF.` and saving the state dict again with the current code. It loads with `strict=True` under the current module names. A correct inference run should not report missing keys for `fusion_s`, `fusion_m`, or `fusion_l`.

In short:

```text
Old checkpoint name: EOLO_50_60.61.pth
Corrected checkpoint name: EOLO_50_60.61_corrected_26.06.30.pth
Only the state dict key names were updated. The learned weights are the same.
```

If you already downloaded the older checkpoint and want to convert it yourself, the remap is:

```python
import torch

old_ckpt = torch.load("EOLO_50_60.61.pth", map_location="cpu")
new_ckpt = {k.replace(".AFNet.", ".SREF."): v for k, v in old_ckpt.items()}
torch.save(new_ckpt, "EOLO_50_60.61_key_remapped.pth")
```

For the current released `EOLO_50_60.61_corrected_26.06.30.pth`, this conversion has already been done and verified.

## Citation

If you find our work useful, please consider citing:

```
@article{cao2023chasing,
  title={Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Guided by an Event Camera},
  author={Cao, Jiahang and Zheng, Xu and Lyu, Yuanhuiyi and Wang, Jiaxu and Xu, Renjing and Wang, Lin},
  journal={arXiv preprint arXiv:2309.09297},
  year={2023}
}
```

## Acknowledgements & Contact
We thank the authors ([PyTorch_YOLO-Family](https://github.com/yjh0410/PyTorch_YOLO-Family)) for their open-sourced codes.

For any help or issues of this project, please contact [Jiahang Cao](jcao248@connect.hkust-gz.edu.cn).
