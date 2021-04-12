# STPAN: Learning spatial and spatio-temporal pixel aggregations for image and video denoising
This is a official implementation of the paper *Learning spatial and spatio-temporal pixel aggregations for image and video denoising*. 
The pytorch implementation is still under preparation and will be released soon. 

**Please see [tf branch](https://github.com/jojo23333/STPAN/tree/tf) for original implementation and models.**

**Thanks to z-bingo for re-implementing our method in https://github.com/z-bingo/Deformable-Kernels-For-Video-Denoising.git**

## TODOs
- [x] Release SPAN and PAN code
- [ ] Release Test and Training Scripts
- [ ] Release Datasets
- [ ] Release Video Denoise models
- [ ] Release Image Denoise models

## Installation
For environment setting up:
```
pip install -r requirement.txt
```

## Dataset Preparation

<!-- For testing data: please arange the test frames as the following format:

-path_to_training_set
    -source
        -vid1
            -vid1_000.png
            -vid1_001.png
            -vid1_002.png
            ...
        -vid2
        ... -->
## Training

## Testing
<!-- In the yaml config file:

modify:
DATA.PATH_TO_TEST_SET: "path to your prepared test set"  
TEST.CHECKPOINT_FILE_PATH: "path to downloaded checkpoint"  
TEST.OUTPUT_DIR: "path to your output dir"   -->

### Citation
If you find this repo helpful in your research, please cite our paper.
```
@article{stpan,
  title={Learning spatial and spatio-temporal pixel aggregations for image and video denoising},
  author={Xu, Xiangyu and Li, Muchen and Sun, Wenxiu and Yang, Ming-Hsuan},
  journal={IEEE Transactions on Image Processing},
  year={2020}
}
@article{dcn_denoise,
  title={Learning deformable kernels for image and video denoising},
  author={Xu, Xiangyu and Li, Muchen and Sun, Wenxiu},
  journal={arXiv preprint arXiv:1904.06903},
  year={2019}
}
```
Contact: muchenli1997@gmail.com
