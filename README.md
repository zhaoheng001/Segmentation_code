# Segmentation_code
## System Requirements
### OS Requirements
This code is supported for macOS and Linux. The package has been tested on the following systems:
- macOS: Monterey (12.6.8)
- Linux: Ubuntu 16.04
- Python 3.9.16

### Python dependencies and installation
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



## Demo

### Training and test dataset
- unzip `\data\train_test_data.zip`
- dataset:

| dataset  | train | test |
|:--------------- |----|----:|
| # |  280 | 70 |

U-Net need fixed input size. So we rescale images and masks to 256x256.

- `data/image_train_256`: train images rescaled to 256x256
- `data/mask_train_256`: train masks rescaled to 256x256
- `data/image_test_256`: test images rescaled to 256x256
- `data/mask_test_256`: test masks rescaled to 256x256

### Download pre-trained model

- Download model `Model_GESU_oct16.hdf5` from https://www.dropbox.com/scl/fi/1twwtxwpksqy43x25739k/Model_GESU_oct16.hdf5?rlkey=dybz3tun4etuayv1q994gudpg&dl=0

### Run train and test script
Edit and run `train_test_demo.ipynb`:
- set data and model path:
  - `train_path`, `train_label`: local training images and masks.
  - `test_path`, `test_label`: local test images and masks.
  - `model_path` : load pre-trained model path.
- training part:
```python
>>> GESU_net = myGESUnet(img_rows = 256, img_cols= 256, train_path=train_path, train_label=train_label, test_path=test_path, test_label=test_label)
>>> GESU_net.load_data()
>>> GESU_net.train(epochs, batches, model_path)
```
If you use the pre-trained model, just commented out `GESU_net.train(epochs, batches, model_path)` and then run test part
- test part:
```python
>>> model = myGESUnet(img_rows = 256, img_cols= 256, train_path=train_path, train_label=train_label, test_path=test_path, test_label=test_label)
imgs_train, imgs_mask_train, imgs_test = model.load_data()
>>> model.load_weights(os.path.join(model_path, "Model_GESU_oct16.hdf5"))
>>> imgs_mask_test = model.predict(imgs_test[:,:,:,0], batch_size=1, verbose=1)
>>> np.save('imgs_mask_test.npy', imgs_mask_test)
```

## Examples
![example1](https://github.com/zhaoheng001/Segmentation_code/blob/main/results/result1.png)
![example2](https://github.com/zhaoheng001/Segmentation_code/blob/main/results/result2.png)
![example3](https://github.com/zhaoheng001/Segmentation_code/blob/main/results/result3.png)



