import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.unet import Unet,U_Net,UNet_2Plus,UNet3Plus,AttU_Net,R2AttU_Net,R2U_Net,DeepLab,Unetnew
from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# --------------------------------------------#
# There are 2 parameters that need to be changed to predict using your own trained models
# model_path and num_classes both need to be modified!

# --------------------------------------------#
class Unet(object):
    _defaults = {
        # -------------------------------------------------------------------#
        # model_path points to the weights file in the logs folder.
        # After training, there are multiple weights in the logs folder, just choose the one with lower loss in the validation set.
        # Lower validation set loss does not mean miou is higher, it only means that the weight has better generalization performance on the validation set.
        # -------------------------------------------------------------------#
        "model_path": r'D:\bianyiqi\pycharm\unet\logs\Unet\ep100-loss0.118-val_loss0.125.pth',
        # --------------------------------#
        #   Number of classes to be distinguished +1
        # --------------------------------#
        "num_classes": 2,
        # --------------------------------#
        # Backbone networks used: vgg, resnet50
        # --------------------------------#
        "backbone": "vgg",
        # --------------------------------#
        # Enter the size of the image
        # --------------------------------#
        "input_shape": [256, 256],
        # -------------------------------------------------#
        # The mix_type parameter is used to control how the test results are visualized
        # mix_type
        # mix_type = 0 means that the original image is mixed with the generated image.
        # mix_type = 1 means that only the generated image is preserved
        # mix_type = 2 means that only the background is removed and only the targets in the original image are kept.
        # -------------------------------------------------#
        "mix_type": 0,
        # --------------------------------#
        # Whether to use Cuda
        # No GPU can be set to False
        # --------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    # Initialize UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        # Picture frames set to different colors
        # ---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # ---------------------------------------------------#
        # Getting the model
        # ---------------------------------------------------#
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        ######################################################################
        ######################################################################
        #self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        #self.net = UNet_2Plus(img_ch=3, num_classes=self.num_classes)
        #self.net = UNet3Plus(n_channels=3, n_classes=self.num_classes, bilinear=True, feature_scale=4, is_deconv=True,is_batchnorm=True)
        self.net = U_Net(img_ch=3, output_ch=self.num_classes)
        #self.net = AttU_Net(img_ch=3, output_ch=self.num_classes)
        #self.net = R2U_Net(img_ch=3, output_ch=self.num_classes, t=2)
        #self.net = R2AttU_Net(img_ch=3, output_ch=self.num_classes, t=2)
        #self.net = DeepLab(num_classes=self.num_classes, backbone="xception", downsample_factor=8, pretrained=False)
        #self.net = Unetnew(num_classes=self.num_classes, backbone=self.backbone)
        ######################################################################
        ######################################################################
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    # Detecting pictures
    # ---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        # ---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent the grayscale image from reporting errors in the prediction.
        # The code only supports the prediction of RGB images, all other types of images are converted to RGB.
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        # Make a backup of the input image to be used later for plotting.
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        # Add gray bars to the image to achieve undistorted resize
        # Can also resize directly for recognition
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        # Add on the batch_size dimension
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            # Images to the web for prediction
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            # Fetch the type of each pixel point
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            # Cut off the gray bar
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            # Perform image resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        # ---------------------------------------------------------#
        #   count
        # ---------------------------------------------------------#
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   Converting a new image into Image form
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   Blend the new image with the original image
            # ------------------------------------------------#
            # image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   Converting a new image into Image form
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            # ------------------------------------------------#
            #   Converting a new image into Image form
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image