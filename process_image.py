

import numpy as np
import cv2
import imutils
import pytesseract
import re
from Vehicle_Register_Manager import *
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def process_image(filename):
    print(filename)
    image = cv2.imread(filename)

    # Resize the image
    image = imutils.resize(image, 500, 500)

    # Show the original image
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(2)

    # RGB to Gray scale conversion
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    blur = cv2.bilateralFilter(gray_img, 11, 17, 17)

    # Find Edges of the grayscale image
    edged = cv2.Canny(blur, 170, 200)

    # Find contours based on Edges
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    # Masking the part other than the number plate
    mask = np.zeros(gray_img.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Character Segmentation
    # Crop the image
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = new_image[topx: bottomx + 1, topy:bottomy + 1]
    cv2.imshow('Crop image', Cropped)
    cv2.waitKey(5)

    # Character Recognition
    # Print Number and remove spaces
    vehicle_number = pytesseract.image_to_string(Cropped, config='--psm 11')

    if (len(vehicle_number) == 0):
        print("number not found")
    else:
        vehicle_number = vehicle_number.replace(" ", "").strip().upper()

        # remove non alphanumerics(ie. garbage special values)
        vehicle_number = re.sub("[^0-9a-zA-Z]+", "", vehicle_number)

        print("Number plate detected is:", vehicle_number)
        print(len(vehicle_number))

        # update register in Vehicle_Register_manager.py
        vehicle_entry(vehicle_number)

        #check if vehicle is authorized in Vehicle_Register_manager.py
        validate_vehicle(vehicle_number)

