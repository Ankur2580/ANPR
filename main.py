import cv2
from process_image import *
import random, string

path = "F:\codebase\ANPR\input\\"

def capture_image():

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("ANPR-WINDOW")

    while True:
        ret, frame = cam.read()
        cv2.imshow("ANPR-WINDOW", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Thank You for using ANPR")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = path + "{}.jpeg".format(get_random_number())
            cv2.imwrite(img_name, frame)

            process_image(img_name)

    cam.release()
    cv2.destroyAllWindows()

    # naming the image randomly
def get_random_number():
    x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
    return (x)

capture_image()
