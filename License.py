import numpy as np
import cv2
import imutils
#import picamera
import pytesseract
import re
import csv
from datetime import datetime
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# License plate detection

#Capturing photo and storing in folder
#with picamera.PiCamera () as camera :
   # camera.resolution = (1280,720)
    #camera.capture("home/pi/Projects/ANPR/images/car.jpeg")

# Read image from file
path = "F:\codebase\ANPR\Photos\\"

#image = cv2.imread(path+'CG04MF2250.jpeg')
image = cv2.imread(path+'car-embossed.jpeg')
#image = cv2.imread(path+'HR26DK8337.jpeg')
#image = cv2.imread(path+'KL40L5577.jpeg') #negative result
#image = cv2.imread(path+'MH12DE1433.jpeg')
#image = cv2.imread(path+'TN01AS9299.jpeg')
#image = cv2.imread(path+'TN37CS2765.jpeg') #negative result



# Resize the image
image = imutils.resize(image, 500,500)

# Show the original image
cv2.imshow("Original Image", image)
cv2.waitKey(2)

# RGB to Gray scale conversion
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("1 - Grayscale Conversion", gray_img)
# cv2.waitKey()


# Noise removal with iterative bilateral filter(removes noise while preserving edges)
blur = cv2.bilateralFilter(gray_img, 11, 17, 17)
# cv2.imshow("2 - Bilateral Filter", blur)
# cv2.waitKey()

# Find Edges of the grayscale image

edged = cv2.Canny(blur, 170, 200)
# cv2.imshow("3- Canny Edges", edged)
# cv2.waitKey()

# Find contours based on
# Edges
contours = cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over contours 
for c in contours:

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

# cv2.imshow('4-Countours', image)
# cv2.waitKey()

# Masking the part other than the number plate
mask = np.zeros(gray_img.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(image,image,mask=mask)

# cv2.imshow('Number plate', new_image)
# cv2.waitKey()

# Character Segmentation

# Crop the image
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = new_image[topx: bottomx+1, topy:bottomy+1]
cv2.imshow('Crop image',Cropped)
cv2.waitKey()

# Character Recognition

# Print Number and remove spaces

vehicle_number = pytesseract.image_to_string(Cropped, config='--psm 11')

if(len(vehicle_number)==0):
    print("number not found")
else :
    vehicle_number = vehicle_number.replace(" ", "").strip().upper()

    #remove non alphanumerics(ie. garbage special values)
    vehicle_number = re.sub("[^0-9a-zA-Z]+", "", vehicle_number)

    print("Number plate detected is:", vehicle_number)

    #enter vehile number, date, time in register
    text_file = open("vehichle_register.txt", "a")
    text_file.write(vehicle_number)
    text_file.write(", ")
    text_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    text_file.write("\n")
    text_file.close()

    #Validate vehicle from records/database

    # reading csv file
    with open("Records.txt", 'r') as csvfile:

        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        #including only first column consisting of numberplates
        included_cols = [0]
        for row in csvreader:
            content = list(row[i] for i in included_cols)
            # Checking if detected numberplate exists in the records
            if( vehicle_number in content):
                print("Authorized Vehicle")
            else:
                print("Unauthorized vehicle")
                break;

