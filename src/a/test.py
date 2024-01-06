import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from labspace import singleTransfer
import pandas as pd

inputImg=(cv2.imread("../data/inputs/examples/0006_001.png",cv2.IMREAD_COLOR))
inputImg= np.float32(cv2.cvtColor(inputImg, cv2.COLOR_BGR2Lab))

refImg=(cv2.imread("../data/inputs/examples/2910450431_56f1c774ed_z.png",cv2.IMREAD_COLOR))
refImg=np.float32(cv2.cvtColor(refImg, cv2.COLOR_BGR2Lab))

refImg_bg=np.float32(cv2.imread("../data/inputs/examples/2910450431_56f1c774ed_z.jpg",cv2.IMREAD_COLOR))

inputImg_mask = cv2.imread("../data/inputs/examples/0006_001_mask.png")

refImg_mask = cv2.imread("../data/inputs/examples/2910450431_56f1c774ed_z_mask.png",0)

style_lm = np.array(pd.read_csv('../data/inputs/examples/2910450431_56f1c774ed_z.lm', header=None),dtype='float32')

input_lm = np.array(pd.read_csv('../data/inputs/examples/0006_001.lm', header=None),dtype='float32')


########## Style Transfer ##########
inputImg_transformed = inputImg.copy()
m1, m2 = (np.min(inputImg_transformed[:,:,0]), np.max(inputImg_transformed[:,:,0]))
inputImg_transformed[:,:,0]=singleTransfer(inputImg[:,:,0], refImg[:,:,0], style_lm, input_lm)
inputImg_transformed[:,:,0] = (inputImg_transformed[:,:,0]-np.min(inputImg_transformed[:,:,0]))*(m2-m1)/(np.max(inputImg_transformed[:,:,0])-np.min(inputImg_transformed[:,:,0])) + m1
inputImg_transformed_u8 = inputImg_transformed.astype('uint8')
inputImg_transformed_u8_rgb = cv2.cvtColor(inputImg_transformed_u8.copy(), cv2.COLOR_Lab2RGB)

### background replacement
inputImg_mask = inputImg_mask/255
tempOut = np.float32(inputImg_transformed_u8_rgb)
outImg = inputImg_mask*tempOut + (1-inputImg_mask)*refImg_bg
outImg = outImg.astype('uint8')
outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2GRAY)

cv2.imwrite('../data/outs/style_transfered_output.png',outImg)

###### plotting outputs
refImgrgb=cv2.cvtColor(refImg.astype(np.uint8),cv2.COLOR_Lab2RGB)
inputImgrgb=cv2.cvtColor(inputImg.astype(np.uint8),cv2.COLOR_Lab2RGB)

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1, 3, 1)
plt.imshow(refImgrgb)
fig.add_subplot(1, 3, 2)
plt.imshow(outImg,cmap='gray')
# plt.imshow(cv2.cvtColor(outImg, cv2.COLOR_BGR2GRAY),cmap='gray')
# plt.imshow(inputImg_transformed_u8_rgb)
fig.add_subplot(1, 3, 3)
plt.imshow(inputImgrgb)
plt.savefig('../data/outs/style_transfered_output_comparison.jpg')
