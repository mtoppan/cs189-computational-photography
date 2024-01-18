import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from hdr import readImagesAndExposures, runRadiometricCalibration, mergeExposureStack, tonemap_photographic, gamma_correction, XYZ2xyY, xyY2XYZ
from cp_hw2 import lRGB2XYZ, XYZ2lRGB, read_colorchecker_gm
from cp_exr import writeEXR
import cv2

### This code does not include all the necessary variables as it was transplanted over here
### in order to demonstrate that an effort was made to conduct whitebalancing as suggested, by
### evening out the grayscale values. However, it was not successful (and instead turne the image
### entirely gray)

### Based on the color correcting workflow as well as manual whitebalancing.
white_clicks = color_clicks[:, 18:24]
white_im = hdr_color[white_clicks[0], white_clicks[1], :]
white_im = np.hstack((white_im, np.ones((6,1))))

white_range = np.reshape(np.linspace(0, 1, 6), (6, 1))
white_range = np.hstack((white_range, white_range, white_range, np.ones((6,1))))

wb = np.linalg.lstsq(white_im, white_range, rcond=None)[0]

averages = np.mean(wb, axis=0)
print(averages)
averages = averages[:3] / averages[3]
r = averages[0]
g = averages[1]
b = averages[2]

red = hdr_color[:, :, 0] / r
green = hdr_color[:, :, 1] / g
blue = hdr_color[:, :, 2] / b

hdr_wb = np.dstack((red, green, blue))

plt.imshow(gamma_correction(hdr_wb / np.max(hdr_wb)))
plt.show()