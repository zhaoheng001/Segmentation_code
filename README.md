# Segmentation_code

## pre-trained model
Download pret-rained model `Model_GESU_oct17.hdf5` from https://www.dropbox.com/s/d003z64vck9gtjp/Model_GESU_oct17.hdf5?dl=0

## train and test script
Edit and run `train_config.ipynb`:

 - `train_path`, `train_label`: local training images and masks.
 - `test_path`, `test_label`: local test images and masks.
 - `model_path` : load pre-trained mmodel path.

## dataset

| dataset  | train | test |
|:--------------- |----|----:|
| # |  280 | 70 |

- `data/image_train_256`: train images rescaled to 256x256
- `data/mask_train_256`: train masks rescaled to 256x256
- `data/image_test_256`: test images rescaled to 256x256
- `data/mask_test_256`: test masks rescaled to 256x256
