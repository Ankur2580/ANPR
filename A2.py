import numpy as np
import cv2
import imutils
import pytesseract
# License plate detection

# Read image from file
image = cv2.imread('car.jpeg')

# Resize the image
image = imutils.resize(image, width=500)

# Show the original image
cv2.imshow("Original Image", image)
cv2.waitKey()

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

# Find contours based on Edges
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

# Print Number

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Number plate detected is:", text)
