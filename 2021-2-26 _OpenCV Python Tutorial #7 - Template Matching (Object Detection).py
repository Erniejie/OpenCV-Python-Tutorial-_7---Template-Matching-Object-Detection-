#OpenCV Python Tutorial #7 - Template Matching (Object Detection)
#Tech With Tim - Uploaded 21 Feb 2021

import numpy as np
import cv2

img = cv2.resize(cv2.imread("soccer_practice.jpg",0),(0,0),fx=0.8,fy=0.8)  # 0: grayscale colour
#template = cv2.imread("ball.png",0)
template = cv2.resize(cv2.imread("shoe.png",0),(0,0),fx=0.8,fy=0.8)
h,w =template.shape

# Template Macthing methods

methods = [cv2.TM_CCOEFF,cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED,cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    #comment: sometimes min_loc or max_loc, gives the best solution

    #Correct Location
    bottom_right = (location[0] +w,location[1] + h)
    cv2.rectangle(img2,location,bottom_right,255,5)
    cv2.imshow("Match",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #(W-w+1,H-h+1)  # w: width of template image, W:width of base image
    
    
    


