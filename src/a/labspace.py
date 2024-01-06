import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from WrapOperator import wrapOperator
import pandas as pd


noOfLevels=6

def lapStack(inputImg):
    h,w=np.shape(inputImg)
    lapStack=np.empty((h,w,noOfLevels))

    lap=[]
    for i in range(0,noOfLevels):
        sigma2 = 2**(i+1)
        ksize2 = 6*sigma2-1
        
        gaBlur2=cv2.GaussianBlur(inputImg.copy(),(ksize2,ksize2),sigma2,sigma2)
        if(i==0):
            lap.append(inputImg-gaBlur2)
        else:
            sigma1 = 2**i
            ksize1 = 6*sigma1-1
            gaBlur1=cv2.GaussianBlur(inputImg.copy(),(ksize1,ksize1),sigma1,sigma1)
            lap.append(gaBlur1-gaBlur2)
            if(i== noOfLevels-1):
                lap.append(gaBlur2)
    return lap

def localEnergyStack(gausStack):
    S=[]
    #gausStack=np.array(gausStack)
    #gausStackSq=np.square(gausStack)
    gausStackSq=[i**2 for i in gausStack]
    
    for i in range(noOfLevels):
        sigma=2**(i+1)
        # c=0
        # if ((5*sigma)%2==0):c=1
        # # S.append(cv2.GaussianBlur(gausStackSq[i],(5*sigma+c,5*sigma+c),sigma,sigma))
        S.append(cv2.GaussianBlur(gausStackSq[i],(6*sigma-1,6*sigma-1),sigma,sigma))
        # S.append(np.sqrt(cv2.GaussianBlur(gausStackSq[i],(6*sigma-1,6*sigma-1),sigma,sigma)))
    
    return S

def singleTransfer(inputImg,refImg,style_lm,input_lm):
    lapIn=lapStack(inputImg)
    lapRef=lapStack(refImg)

    Si=localEnergyStack(lapIn)
    Sr=localEnergyStack(lapRef)

    ## Wrap function
    
    vy, vx, yy, xx =wrapOperator(style_img=refImg.copy(), input_img=inputImg.copy(), style_lm=style_lm, input_lm=input_lm)

    wrLoRe=[]
    for i in range(noOfLevels):
        tempwarp = np.ones(Sr[i].shape)
        tempwarp[yy.astype(int), xx.astype(int)] = Sr[i][vy, vx]
        wrLoRe.append(tempwarp)

    ##Robust Transfer

    #calculating the gain
    norSi=np.zeros_like(np.array(Si))
    gain=np.zeros_like(np.array(Si))
    for i in range(noOfLevels):
        norSi[i]=Si[i]#/np.max(Si[i])
        gain[i]=np.sqrt(np.array(wrLoRe[i])/(np.array(norSi[i])+0.0001))


    #robust gain
    ##threshold the gain bw 0.9 to 2.8
    gain[gain>2.8]=2.8
    gain[gain<0.9]=0.9

    lapout=[]
    robGain=[]
    for i in range(6):
        sigma2 = 3*(2**(i))
        ksize2 = 6*sigma2-1
        robGain.append(cv2.GaussianBlur(gain[i],(ksize2,ksize2),sigma2,sigma2))
        # robGain.append(cv2.GaussianBlur(gain[i],(5*(3*(2**(7)))+1,5*(3*(2**(7)))+1),3*(2**(7)),(3*(2**(7)))))
        lapout.append(lapIn[i]*robGain[i])

    resOut = np.ones(refImg.shape)
    resOut[yy.astype(int), xx.astype(int)] =lapRef[6][vy, vx]

    ## find the reconstructed output from laplacian pyramids and the residue
    ### acc to paper remove the first 3 high freq. layers
    finReImg=np.zeros_like(lapout[0])
    for i in range(6-1,-1,-1):
    # for i in range(6):
    # for i in range(3,6):
        finReImg+=lapout[i]
    finReImg+=resOut

    
    """ fig = plt.figure(figsize=(20, 20))

    fig.add_subplot(1, 3, 1)
    plt.imshow(refImg,cmap="gray")
    fig.add_subplot(1, 3, 2)
    plt.imshow(finReImg,cmap="gray")
    fig.add_subplot(1, 3, 3)
    plt.imshow(inputImg,cmap="gray")

    plt.show() """
    return finReImg
