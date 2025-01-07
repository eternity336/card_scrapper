import cv2
import numpy as np
import getopt, sys

#--- dimensions of the cropped and aligned card 
card_width = 1500
card_height = 2100
card_area = card_height*card_width
card_forgivness = 10

path = './input/'
output_path = './output/'
input_filename = 'cards.jpg'
output_filename = 'card'
file_num = 0
rot_ang = .7

argumentList = sys.argv[1:]
options = "hi:o:d:r:"
long_options = ["help", "input_file=", "output_file=", "dimensions=", "rotation="]

#-- Functions for application

def get_crop(image, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]

def get_crop_wh(image,x=0,y=0,w=card_width,h=card_height):
    return image[y:y+h, x:x+w]

def get_outline(image, old_cnt):
    #--- convert to grayscale ---
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('im_gray_result', imgray)
    #--- perform Otsu threshold to binarize the image (black and white) ---
    ret2, th2 = cv2.threshold(imgray,220,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #--- only finding the external contours ---
    contours, _ =    cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    if cnt is not None:
        print(cv2.boundingRect(cnt))
        print(cv2.contourArea(old_cnt), cv2.contourArea(cnt), (card_width*card_height))
        # cv2.imshow('im_outline_result', get_crop(image, cnt))
        return cnt
    return old_cnt

def save_image(image, filename = output_filename, file_ext = '.png' ):
    global file_num
    file_num += 1
    cv2.imwrite(output_path + filename + str(file_num) + file_ext, image)
    return

def dist2target_dim(box, target_width=card_width, target_height=card_height):
    """
    Calculates the "distance" between a bounding box's dimensions and target dimensions.

    Args:
        box: A tuple (x1, y1, x2, y2) representing the bounding box.
        target_width: The target width.
        target_height: The target height.

    Returns:
        The Euclidean distance between the box's dimensions and the target dimensions.
    """
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    return np.sqrt((box_width - target_width)**2 + (box_height - target_height)**2)

def test_rotation(im,old_cnt,rot_ang=rot_ang,dir=0):
    # cv2.imshow('test_result', get_crop(im, old_cnt))
    # cv2.waitKey(0)
    x, y, w, h = cv2.boundingRect(old_cnt)
    cropped_image = im[y:y+h, x:x+w]
    (height, width) = cropped_image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix_l = cv2.getRotationMatrix2D(center, -rot_ang, 1.0) # 1.0 is the scale factor
    rotation_matrix_r = cv2.getRotationMatrix2D(center, rot_ang, 1.0) # 1.0 is the scale factor
    im_result_l = cv2.warpAffine(cropped_image, rotation_matrix_l, (width, height))
    im_result_r = cv2.warpAffine(cropped_image, rotation_matrix_r, (width, height))
    cnt_l = get_outline(im_result_l, old_cnt)
    cnt_r = get_outline(im_result_r, old_cnt)
    box_l = cv2.boundingRect(cnt_l)
    box_r = cv2.boundingRect(cnt_r)
    distance1 = dist2target_dim(box_l)
    distance2 = dist2target_dim(box_r)
    
    if (dir == 0):
        print("L and R:", box_l, box_r)
        print("COMP:", distance1,distance2)
        if distance1 < distance2:
            print("Left is closer to the target dimensions.")
            return test_rotation(get_crop(im_result_l, cnt_l),cnt_l,rot_ang,1)
        elif distance2 < distance1:
            print("Right is closer to the target dimensions.")
            return test_rotation(get_crop(im_result_r, cnt_r),cnt_r,rot_ang,2)
        else:
            if (cv2.contourArea(cnt_l) - card_area <= card_forgivness):
                return (get_crop(im_result_l, cnt_l),cnt_l)        
            if (cv2.contourArea(cnt_r) - card_area <= card_forgivness):
                return (get_crop(im_result_r, cnt_r),cnt_r)
            print("Both are equally close to the target dimensions.")
            return test_rotation(im,old_cnt,rot_ang,dir)
    elif(dir == 1):
        if (box_l[2] < card_width or box_l[3] < card_height):
            return (im, old_cnt)
        if (abs(cv2.contourArea(cnt_l) - card_area) <= card_forgivness):
            return (get_crop(im_result_l, cnt_l),cnt_l)
        print("Left.")
        return test_rotation(get_crop(im_result_l, cnt_l),cnt_l,rot_ang,1)
    else:
        if (box_r[2] <= card_width or box_r[3] <= card_height):
            return (im, old_cnt)
        if (abs(cv2.contourArea(cnt_r) - card_area) <= card_forgivness):
            return (get_crop(im_result_r, cnt_r),cnt_r)
        print("Right.")
        return test_rotation(get_crop(im_result_r, cnt_r),cnt_r,rot_ang,2)

def largest_contour(contours):
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop_img = im2[y:y+h, x:x+w]
    get_outline(crop_img,cnt)
    (im_result, cnt_result) = test_rotation(im,cnt,.5)
    return get_crop_wh(im_result)
    

#--- START OF APPLICATION

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
    
    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-h", "--Help"):
            print ("Displaying Help")
            print ("-i --input_file Input file for card extraction.")
            print ("-o --output_file filename for card output.")
            print ("-d --dimensions of final card.")
            print ("-r --rotation How much angle each turn.")
    
            sys.exit(0) 
        elif currentArgument in ("-i", "--input_file"):
            input_filename = currentValue
            print("Input file set to ", path + input_filename)   
        elif currentArgument in ("-o", "--output_file"):
            output_filename = currentValue
            print("Input file set to ", path, output_filename)
        elif currentArgument in ("-d", "--dimensions"):
            try:
                w,h = currentValue.split("x")
                card_height = int(h)
                card_width = int(w)
                card_area = card_width * card_height
            except Exception as e:
                print('Dimensions are not correct try wxh. Example 500x700', e)
        elif currentArgument in ("-r", "--rotation"):
            try:
                rot_ang=float(currentValue)
            except Exception as e:
                print('Rotation is not correct try a float or int. Example .7', e)

    #--- load in image
    try:
        im = cv2.imread(path + input_filename)
        print("Image Loaded")
    except Exception as e:
        print("Error loading Image. ", e)
        sys.exit(0)
    
    #--- convert to grayscale ---
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #--- perform Otsu threshold to binarize the image (black and white) ---
    ret2, th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    im2 = im.copy()

    #--- only finding the external contours ---
    contours, hierarchy =    cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if (cv2.contourArea(cnt) >= card_area):
            x, y, w, h = cv2.boundingRect(cnt)
            crop_img = im2[y:y+h, x:x+w]
            get_outline(crop_img,cnt)
            (im_result, cnt_result) = test_rotation(im,cnt,rot_ang)
            save_image(get_crop_wh(im_result))
        
    # largest_contour(contours)

    cv2.destroyAllWindows()
    print("Complete.")
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
