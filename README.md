# Segmentation_code
train_config.ipynp include train and test process.

Download pret-rained model `Model_GESU_oct17.hdf5` from https://www.dropbox.com/s/d003z64vck9gtjp/Model_GESU_oct17.hdf5?dl=0

## train model
Edit and run `train_config.ipynb`:

 - `train_path`, `train_label`: local training images and masks.
 - `test_path`, `test_label`: local test images and masks.
 - `model_path` : load pre-trained mmodel path.

## dataset

| `trainsamples`  | 10 | 100 | 1.000 | 10.000 |
|:--------------- | --:| ---:| -----:| ------:|
| `--maxshards`   |  1 |  10 |    19 |     28 |
