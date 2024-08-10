"""
数据增强方式:
https://blog.csdn.net/qq_38973721/article/details/128700920
包含好多种数据增强方式的代码
"""

import os
import random
from PIL import Image
import cv2
import numpy as np
from random import randint, uniform


class DataAugmentation:
    def __init__(self,
                 rotation_rate=0.2, change_light_rate=0.1, add_noise_rate=0.1, blur_rate=0.1,
                 saturation_rate=0.1, hue_rate=0.1, exposure_rate=0.1, cutout_rate=0.2,
                 is_addNoise=True, is_changeLight=False, is_saturation=True,
                 is_changeHue=False, is_change_exposure=True, is_cutout=True,
                 is_resize=True, is_blur=True, is_rotation=True,
                 expected_size=224,
                 max_rotation_angle=90, saturation_value=0.4, changeLight_alpha=0.6,
                 noise_var=20, hue_value_min=100, hue_value_max=150, exposure_gamma=0.9,
                 blur_kernel_size=5, cutout_holes_num=3, cutout_hole_length=10):

        self.expected_size = expected_size
        self.rotation_rate = rotation_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.saturation_rate = saturation_rate
        self.hue_rate = hue_rate
        self.exposure_rate = exposure_rate
        self.cutout_rate = cutout_rate
        self.blur_rate = blur_rate

        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_saturation = is_saturation
        self.is_changeHue = is_changeHue
        self.is_change_exposure = is_change_exposure
        self.is_cutout = is_cutout
        self.is_blur = is_blur
        self.is_resize = is_resize
        self.is_rotation = is_rotation

        self.saturation_value = saturation_value
        self.max_rotation_angle = max_rotation_angle
        self.changeLight_alpha = changeLight_alpha
        self.noise_var = noise_var
        self.hue_value_min = hue_value_min
        self.hue_value_max = hue_value_max
        self.exposure_gamma = exposure_gamma
        self.blur_kernel_size = blur_kernel_size
        self.cutout_holes_num = cutout_holes_num
        self.cutout_hole_length = cutout_hole_length

    def addNoise(self, img):
        noise = np.random.normal(0, self.noise_var, img.shape)
        noisy_img = noise + img
        # noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        # 大多数图像库支持数据格式为unit8, 限制在0-255之间, 所以用.astype(np.unit8)
        # 以下代码也可以使用
        noisy_img = cv2.convertScaleAbs(np.clip(noisy_img, 0, 255))
        return noisy_img

    def change_light(self, img):
        alpha = random.uniform(self.changeLight_alpha, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    def chenge_hue(self, img):
        img = cv2.convertScaleAbs(img)
        hue_value = random.randint(self.hue_value_min, self.hue_value_max)
        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Update the hue value of the image
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_value) % 180

        # Convert the image back to BGR
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image

    def saturation(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * self.saturation_value
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def change_exposure(self, img):
        gamma_img = np.power(img / 255, self.exposure_gamma)
        return cv2.convertScaleAbs(255 * gamma_img)

    def blur(self, img):
        return cv2.blur(img, (self.blur_kernel_size, ) * 2)

    def cutout(self, img):
        """
        https://github.com/uoguelph-mlrg/Cutout
        """
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.float32)

        for n in range(self.cutout_holes_num):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.cutout_hole_length // 2, 0, h)
            y2 = np.clip(y + self.cutout_hole_length // 2, 0, h)
            x1 = np.clip(x - self.cutout_hole_length // 2, 0, w)
            x2 = np.clip(x + self.cutout_hole_length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0

        mask = np.stack([mask] * 3, axis=-1)  # 如果这句语句在循环体内, 就持续堆叠了
        img = (mask * img).astype(np.uint8)

        return img

    def resize(self, img):
        return cv2.resize(img, (self.expected_size, ) * 2)

    def rotation(self, img):
        center = (img.shape[1] // 2, img.shape[0] // 2)
        angle = random.randint(0, self.max_rotation_angle)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, center)
        return img


    def augmentation(self, img):
        if self.is_change_exposure and uniform(0, 1) < self.exposure_rate:
            img = self.addNoise(img)
        if self.is_blur and uniform(0, 1) < self.blur_rate:
            img = self.blur(img)
        if self.is_cutout and uniform(0, 1) < self.cutout_rate:
            img = self.cutout(img)
        if self.is_saturation and uniform(0, 1) < self.saturation_rate:
            img = self.saturation(img)
        if self.is_addNoise and uniform(0, 1) < self.add_noise_rate:
            img = self.addNoise(img)
        if self.is_changeHue and uniform(0, 1) < self.hue_rate:
            img = self.chenge_hue(img)
        if self.is_changeLight and uniform(0, 1) < self.change_light_rate:
            img = self.change_light(img)
        if self.is_resize:
            img = self.resize(img)

        return img

if __name__ == "__main__":
    need_data_aug = 2
    DA = DataAugmentation()

    source_img_path = '../dataset/Rename'
    save_img_path = '../dataset/valid_ImageDataset'

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    for root, dirs, files in os.walk(source_img_path):
        # 以下是用来创建新文件夹的
        for dirs_name in dirs:
            save_dir = os.path.join(save_img_path, dirs_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                save_dir = os.path.join(save_img_path, os.path.basename(root))
                file_prefix, file_suffix = os.path.splitext(file)

                for i in range(need_data_aug):
                    aug_img = DA.augmentation(img)
                    aug_img_name = f"{file_prefix}_{i+1}{file_suffix}"
                    cv2.imwrite(os.path.join(save_dir, aug_img_name), aug_img)
                    print(f"Image Saved to {os.path.join(save_dir, aug_img_name)}")



