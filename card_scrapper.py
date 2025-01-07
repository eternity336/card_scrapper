import cv2
import numpy as np
from PIL import Image
from functools import reduce
from operator import mul
import getopt, sys
from collections import Counter

#--- dimensions of the cropped and aligned card 
dpi = 300
card_width = 750
card_height = 1050
card_area = card_height*card_width
card_forgivness = 10

path = './input/'
output_path = './output/'
input_filename = 'cards.jpg'
output_filename = 'card'
file_num = 0

argumentList = sys.argv[1:]
options = "hi:o:d:b"
long_options = ["help", "input_file=", "output_file=", "dimensions=", "bg-white"]

#-- Functions for application
def get_lap(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return np.uint8(np.absolute(lap))

def get_sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    return cv2.bitwise_or(sobelx, sobely)
 
def get_canny(image):
    return cv2.Canny(image, 155, 175)

def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,199, 5)

def get_dpi(filename, min_dpi=200):
    global dpi
    global card_width
    global card_height
    global card_area
    try:
        im = Image.open(filename)
        dpi_info = im.info.get('dpi')
        if dpi_info:
            dpi = round(sum(dpi_info)/2)
        else:
            print("DPI information not found in the image metadata.")
    except FileNotFoundError:
        print("Error file not found.")
        sys.exit()
    if dpi < min_dpi:
        dpi = min_dpi
    card_width = dpi * 2.5
    card_height = dpi * 3.5
    card_area = card_height * card_width
    print(f"Image DPI: {dpi}")
    print(f"Image HxW: {card_height}x{card_width}")
    print(f"Card Area: {card_area}")

def set_dpi(_dpi):
    global dpi
    global card_width
    global card_height
    global card_area
    dpi = _dpi
    card_width = dpi * 2.5
    card_height = dpi * 3.5
    card_area = card_height * card_width
    print(f"Image DPI: {dpi}")
    print(f"Image HxW: {card_height}x{card_width}")
    print(f"Card Area: {card_area}")

def drawRectangle(frame, bbox, thickness=1):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), thickness, 1)

def drawBoundingBox(image, box, thickness=1):
    box = np.intp(box)
    cv2.drawContours(image, [box],0,(255, 0, 0),thickness)

def get_crop(image, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]

def get_crop_wh(image,x=0,y=0,w=card_width,h=card_height):
    return image[y:y+h, x:x+w]

def get_outline(image, old_cnt):
    #--- convert to grayscale ---
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #--- perform Otsu threshold to binarize the image (black and white) ---
    ret2, th2 = cv2.threshold(imgray,220,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #--- only finding the external contours ---
    contours, _ =    cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    if cnt is not None:
        print(cv2.boundingRect(cnt))
        print(cv2.contourArea(old_cnt), cv2.contourArea(cnt), (card_width*card_height))
        return cnt
    return old_cnt

def save_image(image, filename = output_filename, file_ext = '.png' ):
    global file_num
    file_num += 1
    cv2.imwrite(output_path + filename + str(file_num) + file_ext, image)
    return

def getSubImageAndRotate(base_src, cnt, rect):
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image = base_src[y:y+h, x:x+w]
    (height, width) = cropped_image.shape[:2]
    center = (width // 2, height // 2)

    # Get center, size, and angle from rect
    _, size, theta = rect
    
    # Convert to int 
    size = tuple(map(int, [card_width, card_height]))
    print("Size", size)
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    print("M:", M)
    dst = cv2.warpAffine(cropped_image, M, (width, height))
    out = cv2.getRectSubPix(dst, size, center)
    return out

def show_image(image, grey=True):
    if grey:
        cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    else:
        cv2.imshow("", image)
    cv2.waitKey(0)

def get_contours(image, canvas):
    shapes = []
    if len(image.shape) == 2:
        #--- only finding the external contours ---
        tmp_canvas = canvas.copy()
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            cnt_area = cv2.contourArea(c)
            if (cnt_area > card_area * .9 and cnt_area < card_area * 1.1):
                print(card_area, cv2.contourArea(c))
                shapes.append(c)
                min_rect = cv2.minAreaRect(c)
                print("Info: ", min_rect)
                if min_rect[2] > 80:
                    min_rect = (min_rect[0],min_rect[1],min_rect[2]-90)
                drawBoundingBox(tmp_canvas,cv2.boxPoints(min_rect),20)
                show_image(getSubImageAndRotate(canvas, c, min_rect),False)
                save_image(getSubImageAndRotate(canvas, c, min_rect))

        print(len(shapes))
        return tmp_canvas
    else:
        print(len(shapes))
        return image

def get_threshold(image, white_bg=False):
    if white_bg:
        return cv2.threshold(cv2.bitwise_not(image),1,255,0)[1]
    else:
        return cv2.threshold(image,140,255,0)[1]   

def get_background_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flatten the image to a 1D array
    pixels = image.reshape((-1, 3))
    # Find the most common color
    color_counts = Counter(tuple(pixel) for pixel in pixels)
    most_common_color = color_counts.most_common(1)[0][0]
    return most_common_color

#--- START OF APPLICATION
try:
    # Parsing argument
    set_dpi = True
    bg = False
    arguments, values = getopt.getopt(argumentList, options, long_options)
    
    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-h", "--Help"):
            print ("Displaying Help")
            print ("-b --bg-white Background of image is white.")
            print ("-d --dpi of card.")
            print ("-i --input_file Input file for card extraction.")
            print ("-o --output_file filename for card output.")
            sys.exit(0) 
        elif currentArgument in ("-i", "--input_file"):
            input_filename = currentValue
            print("Input file set to ", path + input_filename)   
        elif currentArgument in ("-o", "--output_file"):
            output_filename = currentValue
            print("Input file set to ", path, output_filename)
        elif currentArgument in ("-d", "--dpi"):
            try:
                set_dpi = False
                set_dpi(int(currentValue))
            except Exception as e:
                print('DPI is not correct try an int. Example 200', e)
        elif currentArgument in ("-b", "--bg-white"):
            bg = True
       
    #--- load in image
    try:
        im = cv2.imread(path + input_filename)
        if set_dpi:
            get_dpi(path + input_filename)
        print("Image Loaded")
    except Exception as e:
        print("Error loading Image. ", e)
        sys.exit(0)
    
    #--- convert to grayscale ---
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #--- get threshold ---
    result = get_threshold(imgray, bg)
    show_image(im)
    get_contours(result, im)
    cv2.destroyAllWindows()
    print("Complete.")
except Exception as err:
    # output error, and return with an error code
    print ("ERROR: ", str(err))
