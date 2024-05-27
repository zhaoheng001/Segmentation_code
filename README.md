# Segmentation_code
## System Requirements
### OS Requirements
This code is supported for macOS and Linux. The package has been tested on the following systems:
- macOS: Monterey (12.6.8)
- Linux: Ubuntu 16.04
### Python Dependencies
```
keras==2.8.0
matplotlib==3.3.2
numpy==1.26.4
opencv_python==4.5.5.64
Pillow==10.0.0
scikit_learn==1.5.0
scipy==1.13.1
scikit-image==0.22.0
tensorflow==2.8.0
protobuf == 3.20.1
```
### Install dependencies
Dependencies can be installed by `pip install -r requirements.txt`




## pre-trained model
Download pret-rained model `Model_GESU_oct16.hdf5` from https://www.dropbox.com/scl/fi/1twwtxwpksqy43x25739k/Model_GESU_oct16.hdf5?rlkey=dybz3tun4etuayv1q994gudpg&dl=0

## Install requirements
`pip install -r requirements.txt`

## train and test script
Edit and run `train_config.ipynb`:

 - `train_path`, `train_label`: local training images and masks.
 - `test_path`, `test_label`: local test images and masks.
 - `model_path` : load pre-trained model path.

## dataset

| dataset  | train | test |
|:--------------- |----|----:|
| # |  280 | 70 |

U-Net need fixed input size. So we rescale images and masks to 256x256.

- `data/image_train_256`: train images rescaled to 256x256
- `data/mask_train_256`: train masks rescaled to 256x256
- `data/image_test_256`: test images rescaled to 256x256
- `data/mask_test_256`: test masks rescaled to 256x256
