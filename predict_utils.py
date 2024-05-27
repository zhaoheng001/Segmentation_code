import matplotlib.pyplot as plt
import cv2
import os
import glob
import numpy as np
from cv2 import imread
from skimage.morphology import binary_opening, disk, label
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.color import rgb2gray
from PIL import Image
from sklearn.metrics import mean_squared_error
import os, random, shutil

#from skimage.measure import compare_ssim
def to_binary(imgs_path,biimg_path):
    #print(imgs_path)
    imgs = glob.glob(imgs_path + '/*.png')
    #imgs = sorted(imgs, key=lambda name: int(name[50:-4]))
    i = 0
    for file in imgs:
        #print(file)
        img_name = file[file.rindex("/") + 1:]
        #print(img_name)
        img_path = os.path.join(imgs_path,img_name)
        #print(img_path)
        img = cv2.imread(img_path)
        img = np.asarray(img)
        img = img[:,:,0]
        max_res1 = np.max(img)
        binary_img = binary_opening(img>0.2*max_res1, disk(1)).astype(int)
        binary_img = binary_img*255
        #binary_img = Image.fromarray(np.uint8(binary_img))
        #binary_img.save(biimg_path + '%d.png'%(i))
        #print(os.path.join(biimg_path,file))
        cv2.imwrite(os.path.join(biimg_path,img_name), binary_img)
        i = i+1



#calculate the dice loss
def dice(pred_path,label_path):
    pred_imgs = os.listdir(pred_path)
    pred_imgs = sorted(pred_imgs, key=lambda name: int(name[0:-4]))
    recallT = np.zeros(len(pred_imgs))
    precisionT = np.zeros(len(pred_imgs))
    diceT = np.zeros(len(pred_imgs))
    i = 0
    for file in pred_imgs:
        print(file)
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        pred_img = os.path.join(pred_path,file)
        pred = np.array(Image.open(pred_img))
        label_img = os.path.join(label_path,file)
        label = np.array(Image.open(label_img))
        pred = (pred/255).astype(int)
        label = (label/255).astype(int)
        #print(label)
        for y in range(0,256):
            for x in range(0,256):
                if (pred[y,x] == 1 and label[y,x] == 1):
                    true_positive = true_positive + 1
                elif (label[y,x] == 0 and pred[y,x] == 1):
                    false_positive = false_positive + 1
                elif (pred[y,x] == 0 and label[y,x] == 1):
                    false_negative = false_negative + 1
                elif (label[y,x] == 0 and pred[y,x] == 0):
                    true_negative = true_negative + 1
        #precision = true_positive / (true_positive + false_positive)
        #recall = true_positive / (true_positive + false_negative)
        dice = (2*true_positive) / ((2*true_positive) + false_positive + false_negative)
        #print('precision:', precision)
        #print('recall:', recall)
        print('dice loss:', dice)
        #recallT[i] = recall
        #precisionT[i] = precision
        diceT[i] = dice
        i = i + 1
    #recall_mean = np.mean(recallT)
    #precision_mean = np.mean(precisionT)
    dice_mean = np.mean(diceT)
    dice_std = np.std(diceT)
    #print('precision_mean:', precision_mean)
    #print('recall_mean:', recall_mean)
    print('dice_loss_mean:', dice_mean)
    print('dice_loss_std:', dice_std)
    return diceT

#calculate the mse loss
def mse_loss(pred_path,label_path):
    pred_imgs = os.listdir(pred_path)
    pred_imgs = sorted(pred_imgs, key=lambda name: int(name[0:-4]))
    mse_all = np.zeros(len(pred_imgs))
    i = 0
    for file in pred_imgs:
        pred_img = os.path.join(pred_path,file)
        pred = np.array(Image.open(pred_img))
        label_img = os.path.join(label_path,file)
        label = np.array(Image.open(label_img))
        pred = (pred/255).astype(int)
        label = (label/255).astype(int)
        mse_all[i] = mean_squared_error(pred, label)
        i = i + 1
    mse_mean = np.mean(mse_all)
    mse_std = np.std(mse_all)
    print('mse_mean:', mse_mean)
    print('mse_std:', mse_std)
    return mse_all

#show samples of prediction
def show_pred(img_path, label_path, pred_path, num_imgs):
    pred_imgs = os.listdir(img_path)
    pred_imgs = os.listdir(label_path)
    pred_imgs = os.listdir(pred_path)
    sample = random.sample(pred_imgs, num_imgs)
    for file in sample:
        imgs_file = os.path.join(img_path,file)
        labels_file = os.path.join(label_path,file)
        pred_file = os.path.join(pred_path,file)
        imgs = plt.imread(imgs_file)
        labels = plt.imread(labels_file)
        pred = plt.imread(pred_file)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(imgs, cmap='gray')
        plt.title('image')
        plt.subplot(1,3,2)
        plt.imshow(pred, cmap='gray')
        plt.title('VGG 2*UNet')
        plt.subplot(1,3,3)
        plt.imshow(labels, cmap='gray')
        plt.title('ground truth')
    plt.savefig('./%s'%(file))


    
#post process and remove small particles
def post_process(imgs_path,biimg_path):
    imgs = glob.glob(imgs_path + '/*.png')
    #imgs = sorted(imgs, key=lambda name: int(name[50:-4]))
    i = 0
    for file in imgs:
        img_name = file[file.rindex("/") + 1:]
        img_path = os.path.join(imgs_path,img_name)
        #print(img_path)

        img = cv2.imread(img_path, 0)
        kernel_dilation1 = np.ones((3,3), np.uint8)
        kernel_erpsion1 = np.ones((3,3), np.uint8)
        
        img_dilation1 = cv2.dilate(img, kernel_dilation1, iterations=2)
        img_erosion1 = cv2.erode(img_dilation1, kernel_erpsion1, iterations=2)


        # find all of the connected components (white blobs in your image).
        # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img_erosion1)
        # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
        # here, we're interested only in the size of the blobs, contained in the last column of stats.
        sizes = stats[:, -1]
        # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
        # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
        sizes = sizes[1:]
        nb_blobs -= 1

        # minimum size of particles we want to keep (number of pixels).
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
        min_size =256  

        # output image with only the kept components
        im_result = np.zeros((img.shape))
        # for every component in the image, keep it only if it's above min_size
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
                im_result[im_with_separated_blobs == blob + 1] = 255
        


        #kernel_erpsion2 = np.ones((3, 3), np.uint8)
        #kernel_dilation2 = np.ones((3, 3), np.uint8)
        #img_erosion2 = cv2.erode(img_erosion1, kernel_erpsion2, iterations=1)
        #img_dilation2 = cv2.dilate(img_erosion2, kernel_dilation2, iterations=1)



        #binary_img = Image.fromarray(np.uint8(binary_img))
        #binary_img.save(biimg_path + '%d.png'%(i))
        #print(os.path.join(biimg_path,file))
        cv2.imwrite(os.path.join(biimg_path,img_name), im_result)
        i = i+1

def removesmall(imgs_path,biimg_path):
    imgs = glob.glob(imgs_path + '/*.png')
    #imgs = sorted(imgs, key=lambda name: int(name[50:-4]))
    i = 0
    for file in imgs:
        img_name = file[file.rindex("/") + 1:]
        img_path = os.path.join(imgs_path,file)
        #print(img_path)

        img = cv2.imread(img_path, 0)
        # find all of the connected components (white blobs in your image).
        # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img)
        # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
        # here, we're interested only in the size of the blobs, contained in the last column of stats.
        sizes = stats[:, -1]
        # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
        # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
        sizes = sizes[1:]
        nb_blobs -= 1

        # minimum size of particles we want to keep (number of pixels).
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
        min_size =256  

        # output image with only the kept components
        im_result = np.zeros((img.shape))
        # for every component in the image, keep it only if it's above min_size
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
                im_result[im_with_separated_blobs == blob + 1] = 255
        
        cv2.imwrite(os.path.join(biimg_path,img_name), im_result)
        i = i+1


