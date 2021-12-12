from matplotlib import pyplot as plt
import numpy as np
import cv2

class ContourCutter:
    def tester():
        count2 = 0
        contour_limit = 20
        while(count2<40 and len(list_of_files_to_write) != 10):
            count2 += 1
            list_of_files_to_write = []
            contour_limit += 5
            #dst, sorted_ctrs = draw_conts(img_dilated, contour_limit)
            #make_list_of_files(dst, sorted_ctrs, list_of_files_to_write, threshold_area, max_area, count,  "D")

    def __init__(self):
        print()

    def contour(self, image, image_value, iteration):
        img = cv2.imread('/home/jovyan/Python_eksamen/Images/' + image)
        image_for_saving = img
        kernel = np.ones((1,1), np.float32)/1
        img = cv2.filter2D(img, -1, kernel)
   
        img_eroded = cv2.erode(img, kernel, iterations=100)
        #plt.imshow(img)
        #plt.figure()
        #if (billede er lyst -> erode) else (dilate) 

        def draw_conts(img, thresh): 
            kernel = np.ones((1,1), np.float32)/1
            img = cv2.filter2D(img, -1, kernel)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh_H = cv2.threshold(img_grey, thresh, 20, 0)
            _, binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY_INV)
            (_, contours, hierarchy) = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            

            cv2.drawContours(img, contours, -1, (100,255,100), 1)
            plt.imshow(img)
            plt.figure()


            mask = np.ones(img.shape[: 2], dtype = "uint8")
            cv2.drawContours(mask, contours, -1, 0, -1)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(img, img, mask = mask_inv)
            img_fg =  cv2.bitwise_and(img, img, mask = mask)
            #plt.imshow(img)
            dst = cv2.add(img_bg, img_fg)
            return dst, sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        #plt.figure()
        
        contour_limit = 245

        dst, sorted_ctrs = draw_conts(img, contour_limit)
        count = 0
        list_of_files_to_write = []
        threshold_area = 110
        max_area = 500
        
        
        def make_list_of_files(dst, sorted_ctrs, list_of_files, threshold, max, count, type):  
            print(type + " Contours " +str(len(sorted_ctrs))) 
            for c in sorted_ctrs:
                area = cv2.contourArea(c)
                #print("/home/jovyan/Python_eksamen/Images/CutImages/"+str(image_value[count])+"/"+str(image_value[count])+"("+str(iteration)+").jpg" + "  AREA: " + str(area))
                if (area > threshold) and (area < max) :

                    x, y, width, height = cv2.boundingRect(c)
                    img = dst
                    roi = image_for_saving[y: y + height, x: x + width]
                    resized_image = cv2.resize(roi, (28, 28))
                    list_of_files.append(("/home/jovyan/Python_eksamen/Images/CutImages/"+str(image_value[count])+"/"+str(image_value[count])+"("+str(iteration)+type+").jpg", resized_image))
                    #cv2.imwrite("/home/jovyan/Python_eksamen/Images/CutImages/"+str(image_value[count])+"/"+str(image_value[count])+"("+str(iteration)+").jpg", resized_image)
                    count+=1
        
        make_list_of_files(dst, sorted_ctrs, list_of_files_to_write, threshold_area, max_area, count, "NONE")
        count2 = 0
        while (count2<44 and len(list_of_files_to_write) != 10):
            count2 += 1
            list_of_files_to_write = []
            contour_limit -= 5
            dst, sorted_ctrs = draw_conts(img_eroded, contour_limit)
            make_list_of_files(dst, sorted_ctrs, list_of_files_to_write, threshold_area, max_area, count, "E")
        
        count2 = 0
        contour_limit = 20
        while(count2<44 and len(list_of_files_to_write) != 10):
            count2 += 1
            list_of_files_to_write = []
            contour_limit -= 5
            img_copy = cv2.erode(img, kernel, iterations=50)
            img_copy = cv2.dilate(img_copy, kernel, iterations=25)
            dst, sorted_ctrs = draw_conts(img_copy, contour_limit)
            make_list_of_files(dst, sorted_ctrs, list_of_files_to_write, threshold_area, max_area, count, "E&D")
            
        for index, tuples in enumerate(list_of_files_to_write):
            if(len(list_of_files_to_write) == 10):
                cv2.imwrite(tuples[0],tuples[1])


    def contour2(self, image, image_value, iteration):
        img = cv2.imread('/home/jovyan/Python_eksamen/Images/' + image)
        image_for_saving = img
        kernel = np.ones((1,1), np.float32)/1
        img = cv2.filter2D(img, -1, kernel)
   
        img_eroded = cv2.erode(img, kernel, iterations=100)

        def draw_conts(img, thresh): 
            kernel = np.ones((1,1), np.float32)
            img = cv2.GaussianBlur(img,(1,1),0)
            img_eroded = cv2.erode(img, kernel, iterations=6)
            img_grey = cv2.cvtColor(img_eroded, cv2.COLOR_BGR2GRAY)
            ret, thresh_H = cv2.threshold(img_grey, thresh, 20, 0)
            _, binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY_INV)
            (_, contours, hierarchy) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

            cv2.drawContours(img, contours, -1, (0,0,0), 1)
            plt.imshow(img)
            plt.figure()


            mask = np.ones(img.shape[: 2], dtype = "uint8")
            cv2.drawContours(mask, contours, -1, 0, -1)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(img, img, mask = mask_inv)
            img_fg =  cv2.bitwise_and(img, img, mask = mask)
            dst = cv2.add(img_bg, img_fg)
            return dst, sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        contour_limit = 250

        dst, sorted_ctrs = draw_conts(img, contour_limit)
        count = 0
        list_of_files_to_write = []
        threshold_area = 40
        max_area = 500
        
        
        def make_list_of_files(dst, sorted_ctrs, list_of_files, threshold, max, count, image_for_saving):  
            for c in sorted_ctrs:
                area = cv2.contourArea(c)
                if (area > threshold) and (area < max) :
                    x, y, width, height = cv2.boundingRect(c)
                    roi = image_for_saving[y: y + height, x: x + width]
                    resized_image = cv2.resize(roi, (28, 28))
                    list_of_files.append(("/home/jovyan/Python_eksamen/Images/CutImages/"+str(image_value[count])+"/"+str(image_value[count])+"("+str(iteration)+").jpg", resized_image))
                    count+=1
        
        make_list_of_files(dst, sorted_ctrs, list_of_files_to_write, threshold_area, max_area, count, image_for_saving)
        img_length = image_value.split("(")
        img_length = len(img_length[0])
        count2 = 0
        
        while(count2 < 100 and len(list_of_files_to_write) != img_length):
            list_of_files_to_write = []
            count = 0
            count2 += 1
            contour_limit -= 2
            dst, sorted_ctrs = draw_conts(img, contour_limit)
            make_list_of_files(dst, sorted_ctrs, list_of_files_to_write, threshold_area, max_area, count, image_for_saving)

        if(len(list_of_files_to_write) == img_length):
            for index, tuples in enumerate(list_of_files_to_write):
                print(tuples[0])
                cv2.imwrite(tuples[0],tuples[1])
        