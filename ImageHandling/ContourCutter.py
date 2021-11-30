from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import os

class ContourCutter:
    def __init__(self):
        print()
    
    
    
    def cutAndReturn(self, imgname, value, iteration):
        def areaFilter(minArea, inputImage):
            # Perform an area filter on the binary blobs:
            componentsNumber, labeledImage, componentStats, componentCentroids = \
            cv2.connectedComponentsWithStats(inputImage, connectivity = 4)

            # Get the indices / labels of the remaining components based on the area stat
            #(skip the background component at index 0)
            remainingComponentLabels = [i
                for i in range(1, componentsNumber) if componentStats[i][4] >= minArea
            ]

            # Filter the labeled pixels based on the remaining labels,
            # assign pixel intensity to 255(uint8) for the remaining pixels
            filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

            return filteredImage
            
        imagePath = '/home/jovyan/Exam/Images/'
        imageName = imgname

        # Read image:
        inputImage = cv2.imread(imagePath + imageName)
        # Store a copyfor results:
        inputCopy = inputImage.copy()

        # Convert BGR to grayscale:
        grayInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        # Set a lower and upper range for the threshold:
        lowerThresh = 230
        upperThresh = 235

        # Get the lines mask:
        mask = cv2.inRange(grayInput, lowerThresh, upperThresh)

        # Set a filter area on the mask:
        minArea = 50
        mask = areaFilter(minArea, mask)

        # Reduce matrix to a n row x 1 columns matrix:
        reducedImage = cv2.reduce(mask, 1, cv2.REDUCE_MAX)

        # Find the big contours / blobs on the filtered image:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Store the lines here:
        separatingLines = []

        # We need some dimensions of the original image:
        imageHeight = inputCopy.shape[0]
        imageWidth = inputCopy.shape[1]

        # Look for the outer bounding boxes:
        for _, c in enumerate(contours):
            # Approximate the contour to a polygon:
            contoursPoly = cv2.approxPolyDP(c, 3, True)
            # Convert the polygon to a bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)

            # Get the bounding rect 's data: 
            [x, y, w, h] = boundRect

            # Start point and end point:
            lineCenter = y + (0.5 * h)
            startPoint = (0, int(lineCenter))
            endPoint = (int(imageWidth), int(lineCenter))

            # Store the end point in list:
            separatingLines.append(endPoint)

            # Draw the line using the start and end points:
            color = (0, 255, 0)
            cv2.line(inputCopy, startPoint, endPoint, color, 2)

        # Show the image:
        cv2.imshow("inputCopy", inputCopy)
        cv2.waitKey(0)

        # Sort the list based on ascending Y values:
        separatingLines = sorted(separatingLines, key = lambda x: x[1])

        # The past processed vertical coordinate:
        pastY = 0

        # Crop the sections:
        for i in range(len(separatingLines)):
            # Get the current line width and starting y:
            (sectionWidth, sectionHeight) = separatingLines[i]

        # Set the ROI:
        x = 0
        y = pastY
        cropWidth = sectionWidth
        cropHeight = sectionHeight - y

        # Crop the ROI:
        currentCrop = inputImage[y: y + cropHeight, x: x + cropWidth]
        cv2.imshow("Current Crop", currentCrop)
        cv2.waitKey(0)

        # Set the next starting vertical coordinate:
        pastY = sectionHeight

        

        #cv2.imwrite("result1.png", res1)
        #cv2.imwrite(value + '(' + iteration + ')', res2)