from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import os

class ContourCutter:
    def __init__(self):
        print()

    def contour(self, image, image_value, iteration):
        img = cv2.imread('/home/jovyan/Python_eksamen/Images/' + image)
        kernel = np.ones((2,2), np.float32)/4
        img = cv2.filter2D(img, -1, kernel)
        #plt.imshow(img)
        #plt.figure()

        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh_H = cv2.threshold(img_grey, 70, 255, 0)
        (_, contours, hierarchy) = cv2.findContours(thresh_H, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print("Contours " +str(len(contours)))
       
        cv2.drawContours(img, contours, -1, (0,0,0), 1)
        #plt.imshow(img)
        #plt.figure()


        mask = np.ones(img.shape[: 2], dtype = "uint8")
        cv2.drawContours(mask, contours, -1, 0, -1)
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(img, img, mask = mask_inv)
        img_fg =  cv2.bitwise_and(img, img, mask = mask)
        #plt.imshow(img)
        dst = cv2.add(img_bg, img_fg)
        
        #plt.figure()
        count = 0

        threshold_area = 100 
        max_area = 1000 
        sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])   
        for c in sorted_ctrs:
            area = cv2.contourArea(c)         
            if (area > threshold_area) and (area < max_area) :                   
                x, y, width, height = cv2.boundingRect(c)
                img = dst
                roi = img[y: y + height, x: x + width]
                resized_image = cv2.resize(roi, (28, 28))
                cv2.imwrite("/home/jovyan/Python_eksamen/Images/CutImages/"+str(image_value[count])+"/"+str(image_value[count])+"("+str(iteration)+").jpg", resized_image)
                count+=1

