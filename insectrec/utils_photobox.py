import pandas as pd
import numpy as np
seed = 42
np.random.seed(42)
from natsort import natsorted
import os
import cv2
import glob
import matplotlib.pyplot as plt
from nms import non_max_suppression_fast

def get_plate_names_photobox(year='2020', base_dir=None):
    year = f"{year}_photobox"
    if year in ['2020_photobox']:
        print("True")

        # Find plates for all image types known to exist in our dirs
        img_types = ('*.jpg','*.JPG','*.png') 
        all_plates = []        
        files_found = glob.glob(f'{base_dir}/{year}/**/**/*.jpg')
        for p in files_found:
            all_plates.append(p)
        plates = [p for p in all_plates if not p.split('/')[-1].startswith('other') and not p.split('/')[-1].startswith('calibration')]  
        plates = pd.Series(plates).drop_duplicates().tolist()
    else:
        raise ValueError("Wrong year given!")
    return plates

def overlay_image_nms(imgpath, created_data_path, nms_threshold=0.15, plot_orig=True):
    path_annotations = f'{created_data_path}/annotations_photobox/'
    path_images = f'{created_data_path}/images_photobox/'
    path_images_edged = f'{created_data_path}/images_photobox/'
    path_voc_annotations = f'{created_data_path}/voc_annotations_photobox/'
    path_crops_export = f'{created_data_path}/crops_export_photobox/'

    platename = imgpath.split('/')[-1][:-4]

    # Percentage of inner plate to consider (instead of full plate dimensions)
    pct = 80

    img = cv2.imread(imgpath)
    H, W = img.shape[:2]
    hh, ww = round(H*pct/100), round(W*pct/100)
    y, x = round((H-hh)/2), round((W-ww)/2)
    # Getting inner image
    img = img[y:y+hh, x:x+ww]

    red = img[:,:,2].copy() #cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 
    # red = cv2.medianBlur(red, 3)
    edged = cv2.Canny(red, 100,200)

    (cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edged_image = img.copy()
    cv2.drawContours(edged_image, cnts, -1, (0, 255, 0), 1);

    coordinates, contours = [], []
    for cnt in cnts:
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10']/M['m00'])
        except:
            cx = np.nan
        try:
            cy = int(M['m01']/M['m00'])
        except:
            cy = np.nan
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if (20 < area < 900) and (30 < perimeter < 900):
            if np.nan not in [cx, cy]:
                x,y,w,h = cv2.boundingRect(cnt)
                if np.abs(w-h) < 100 and w/h < 5 and h/w < 5:
                    edged_image = cv2.rectangle(edged_image, (x,y), (x+w, y+h), (0,255,0), 2)
                    coordinates.append((x,y,x+w, y+h))
                    contours.append(cnt)

    # load the image and clone it
    boundingBoxes = np.array(coordinates)

    all_bbox_features = []
    bbox_features = {}
    bbox_dim = 120
    bbox_dim_final = 150

    # perform non-maximum suppression on the bounding boxes
    pick, idxs = non_max_suppression_fast(boundingBoxes, nms_threshold)
    # loop over the picked bounding boxes and draw them
    for p, (startX, startY, endX, endY) in enumerate(pick):
        idx = np.where(np.all(pick[p]==boundingBoxes,axis=1))[0][0]
        cnt = contours[idx]
        bbox_features['plate_name'] = platename
        bbox_features['index'] = p+1
        bbox_features['area'] = cv2.contourArea(cnt)
        bbox_features['perimeter'] = cv2.arcLength(cnt, True)
    #     bbox_features['ellipse'] = cv2.fitEllipse(cnt)
        bbox_features['convexity_bool'] = cv2.isContourConvex(cnt)
    #     bbox_features['hull'] =  cv2.convexHull(cnt)
        _, bbox_features['enclosing_circle_radius'] = cv2.minEnclosingCircle(cnt)
        bbox_features['B'] = img[startY:endY+1, startX:endX+1, 0].mean()
        bbox_features['G'] = img[startY:endY+1, startX:endX+1, 1].mean()
        bbox_features['R'] = img[startY:endY+1, startX:endX+1, 2].mean()
        bbox_features['startX'] = startX
        bbox_features['startY'] = startY
        bbox_features['endX'] = endX+1
        bbox_features['endY'] = endY+1
        bbox_features['Class'] = ''

        if startX < bbox_dim or startY < bbox_dim or endX > (ww-bbox_dim) or endY > (hh-bbox_dim):
            continue

        all_bbox_features.append(pd.Series(bbox_features))

        # Save bounding box images
        bbox_center_x = round((startX + endX)/2)
        bbox_center_y = round((startY + endY)/2)

        x_from = round(bbox_center_x - bbox_dim/2)
        x_to = round(bbox_center_x + bbox_dim/2)
        y_from = round(bbox_center_y - bbox_dim/2)
        y_to = round(bbox_center_y + bbox_dim/2)

        bbox_h = y_to - y_from
        bbox_w = x_to - x_from
        assert bbox_h == bbox_dim, "Something wrong with bounding box dimensions"
        assert bbox_w == bbox_dim, "Something wrong with bounding box dimensions"        

        bbox_img = img[y_from:y_to, x_from:x_to].copy()

        # resize the bounding box image
        # If you are enlarging the image, 
        #    you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. 
        # If you are shrinking the image, 
        #    you should prefer to use INTER_AREA interpolation.
        bbox_img_resized = cv2.resize(bbox_img, (bbox_dim_final, bbox_dim_final), interpolation = cv2.INTER_LINEAR)

        cv2.imwrite(f"{path_crops_export}/{platename}_{p+1}.jpg", bbox_img_resized)

    for i, (startX, startY, endX, endY) in enumerate(pick):
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(img, f"{i+1}", (startX-20, startY-10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,250), 4)

    cv2.putText(img, platename, (10, 4150), cv2.FONT_HERSHEY_DUPLEX, 5., (0,0,250), 5)

    cv2.imwrite(f"{path_images_edged}{platename}.jpg",img)

    print(f"{platename}")
    print(f"{len(coordinates)} insects counted in captured image.")
    print(f"{len(pick)} images after non-maximal-suppression\n")

    df = pd.concat(all_bbox_features,axis=1).T
    df.to_csv(f"{path_annotations}{platename}.txt", sep=',')
    
    return img, df
