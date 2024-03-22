import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   If you want to change the color of the corresponding category, go to the __init__ function and change self.colors.
    #-------------------------------------------------------------------------#
    unet = Unet()
    #----------------------------------------------------------------------------------------------------------#
    # mode is used to specify the mode of the test:
    # 'predict' means single picture prediction, if you want to modify the prediction process, such as saving the picture, intercepting the object, etc., you can first see the detailed comments below

    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    # count specifies whether the pixel count (i.e., area) and scale of the target is performed.
    #
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","zuanzao"]
    #----------------------------------------------------------------------------------------------------------#


    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                print(image)
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()


    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
