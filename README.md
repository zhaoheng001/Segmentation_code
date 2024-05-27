# Segmentation_code
## System Requirements
### OS Requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:
- macOS: Monterey (12.6.8)
- Linux: Ubuntu 16.04
### Python Dependencies



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
