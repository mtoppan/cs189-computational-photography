#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from hdr import readImagesAndExposures, runRadiometricCalibration, mergeExposureStack, tonemap_photographic, gamma_correction, XYZ2xyY, xyY2XYZ
from cp_hw2 import lRGB2XYZ, XYZ2lRGB, read_colorchecker_gm, writeHDR
from cp_exr import writeEXR
import cv2

### 1. HDR imaging
## Read images
if not os.path.isdir('../results'):
    os.mkdir('../results')
dir_images = '../data/door_stack/'
nExposure = 16
scaling = 8 # debugging with small-size image
option_input = 'rendered' # ['rendered', 'RAW']

img_list, exposure_list = readImagesAndExposures(dir_images, nExposure, option_input, scaling)
## Run radiometric calibration
l = 5e1 # lambda
option_weight = 'uniform' # ['uniform', 'tent', 'Gaussian', 'photon']
g, w = runRadiometricCalibration(img_list, exposure_list, l, option_weight)
plt.plot(g)
plt.title('Camera response function (log)')
plt.show()
# plt.savefig('../results/camera_response_function.png')

## Merge exposure stack into HDR image
option_merge = 'logarithmic' # ['linear', 'logarithmic']
hdr = mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight)

# ## Write hdr image (.exr and .hdr format)
# hdr = hdr/np.max(hdr)
# brightness = 2**6
# hdr_gamma = gamma_correction(hdr)

writeEXR("../results/hdr_original.exr", gamma_correction(hdr))
writeHDR("../results/hdr_original.hdr", gamma_correction(hdr))

### 2. Color correction and white balancing

# take in image coordinates from ginput

# color_clicks = np.asarray(plt.ginput(24, 120)) 
color_clicks = np.array([[420, 187], [420, 167], [420, 147], [420, 127], [420, 107], [420, 87], [440, 187], [440, 167], [440, 147], [440, 127], [440, 107], [440, 87], [460, 187], [460, 167], [460, 147], [460, 127], [460, 107], [460, 87], [480, 187], [480, 167], [480, 147], [480, 127], [480, 107], [480, 87]])
color_clicks = color_clicks.T

x = color_clicks[0].astype(np.uint64)
x = x.T

y = color_clicks[1].astype(np.uint64)
y = y.T

# convert to rgb coordinates (24 x 3)
# add np.ones to make it 24 x 4
ones = np.ones((24, 1))
coordinates = hdr[y, x, :] #flip x and y?
coordinates = np.hstack((coordinates, ones))

# call in the reference images
r, g, b = read_colorchecker_gm()

# reshape to 24 x 3 and add np.ones for 24 x 4
r = np.reshape(r, (24, 1))
g = np.reshape(g, (24, 1))
b = np.reshape(b, (24, 1))
references = np.hstack((r, g, b, ones))

# np.linalg.lstsq(coordinates, references)[0]
x = np.linalg.lstsq(coordinates, references, rcond=None)[0]

# add np.ones to the original image
hdr_ones = np.ones((hdr.shape[0], hdr.shape[1]))

hdr_homogeneous = np.dstack((hdr, hdr_ones))

# multiply original by least squares result
hdr_mult = hdr_homogeneous @ x
hdr_color = hdr_mult[:, :, :3] / hdr_mult[:, :, 3:4]
hdr_color = np.where(hdr_color < 0, 0, hdr_color)
# normalize

plt.imshow(gamma_correction(hdr_color))
plt.show()

#whitebalance— get the 'white' color range and adjust s.t. the channels are averaged and may be divided by to reach 1.
white_clicks = color_clicks[:, 18:24]

white = hdr_color[white_clicks].T
r = np.average(white[0])
g = np.average(white[1])
b = np.average(white[2])

red = hdr_color[:, :, 0] / r
green = hdr_color[:, :, 1] / g
blue = hdr_color[:, :, 2] / b

hdr_wb = np.dstack((red, green, blue))

plt.imshow(gamma_correction(hdr_wb / np.max(hdr_wb)))
plt.show()

writeEXR("../results/hdr_wb.exr", gamma_correction(hdr_wb / np.max(hdr_wb)))
writeHDR("../results/hdr_wb.hdr", gamma_correction(hdr_wb / np.max(hdr_wb)))

### 3. Photographic tonemapping
# tonemap rgb
K = 0.3
B = 0.8
tm_rgb = tonemap_photographic(hdr_color, K, B)
# tm_rgb = gamma_correction(tm_rgb)
writeEXR("../results/hdr_tonemap_rgb.exr", tm_rgb)
writeHDR("../results/hdr_tonemap_rgb.hdr", tm_rgb)

# tonemap luminance only (xyY)
np.seterr(divide='ignore', invalid='ignore')
xyY = XYZ2xyY(lRGB2XYZ(hdr_color))
xyY[:, :, 2] = tonemap_photographic(xyY[:, :, 2], K, B)
tm_Y = XYZ2lRGB(xyY2XYZ(xyY))
tm_Y = np.clip(tm_Y, 0, None)
writeEXR("../results/hdr_tonemap_Y.exr", tm_Y)
writeHDR("../results/hdr_tonemap_Y.hdr", tm_Y)

# # ### 4. My own HDR image

# # def grayWorldWB(imArray):
# # # • Compute per-channel average.
# #     r_avg = np.mean(imArray[:, :, 0])
# #     g_avg = np.mean(imArray[:, :, 1])
# #     b_avg = np.mean(imArray[:, :, 2])
# # # • Normalize each channel by its average, then by the green average.
# #     r_norm = (imArray[:, :, 0] * g_avg) / r_avg
# #     g_norm = imArray[:, :, 1]
# #     b_norm = (imArray[:, :, 2] * g_avg) / b_avg
    
# #     imArray_WB = np.dstack((r_norm, g_norm, b_norm))
    
# #     return imArray_WB

# # dir_images = '../data/dali_stack/'
# # nExposure = 10
# # scaling = 8 # debugging with small-size image
# # option_input = 'RAW' # ['rendered', 'RAW']

# # img_list, exposure_list = readImagesAndExposures(dir_images, nExposure, option_input, scaling)
# # ## Run radiometric calibration
# # l = 5e1 # lambda
# # option_weight = 'photon' # ['uniform', 'tent', 'Gaussian', 'photon']
# # g, w = runRadiometricCalibration(img_list, exposure_list, l, option_weight)

# # plt.plot(g)
# # plt.title('Camera response function (log)')
# # plt.show()
# # ## Merge exposure stack into HDR image
# # option_merge = 'logarithmic' # ['linear', 'logarithmic']
# # hdr = mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight)
# # hdr = grayWorldWB(hdr)
# # plt.imshow(hdr)
# # plt.show()
# # plt.imshow(gamma_correction(hdr))
# # plt.show()
# # writeEXR("../results/hdrRen.exr", gamma_correction(hdr))
# # writeHDR("../results/hdrRen.hdr", gamma_correction(hdr))

# # K = 0.3
# # B = 0.8
# # tm_rgb = tonemap_photographic(hdr, K, B)
# # plt.imshow(gamma_correction(tm_rgb))
# # plt.show()
# # tm_rgb = gamma_correction(tm_rgb)
# # writeEXR("../results/hdrRenRGB.exr", tm_rgb)
# # writeHDR("../results/hdrRenRGB.hdr", tm_rgb)

# # # tonemap luminance only (xyY)
# # np.seterr(divide='ignore', invalid='ignore')
# # xyY = XYZ2xyY(lRGB2XYZ(hdr))
# # xyY[:, :, 2] = tonemap_photographic(xyY[:, :, 2], K, B)
# # tm_Y = XYZ2lRGB(xyY2XYZ(xyY))
# # tm_Y = np.clip(tm_Y, 0, None)
# # tm_Y = gamma_correction(tm_Y)
# # plt.imshow(tm_Y)
# # plt.show()
# # writeEXR("../results/hdrRenY.exr", tm_Y)
# # writeHDR("../results/hdrRenY.hdr", tm_Y)


# # option_input = 'rendered' # ['rendered', 'RAW']

# # img_list, exposure_list = readImagesAndExposures(dir_images, nExposure, option_input, scaling)
# # ## Run radiometric calibration
# # l = 5e1 # lambda
# # option_weight = 'photon' # ['uniform', 'tent', 'Gaussian', 'photon']
# # g, w = runRadiometricCalibration(img_list, exposure_list, l, option_weight)

# # plt.plot(g)
# # plt.title('Camera response function (log)')
# # plt.show()
# # ## Merge exposure stack into HDR image
# # option_merge = 'linear' # ['linear', 'logarithmic']
# # hdr = mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight)
# # hdr = grayWorldWB(hdr)
# # plt.imshow(hdr)
# # plt.show()
# # plt.imshow(gamma_correction(hdr))
# # plt.show()
# # writeEXR("../results/hdrRen.exr", gamma_correction(hdr))
# # writeHDR("../results/hdrRen.hdr", gamma_correction(hdr))

# # K = 0.3
# # B = 0.8
# # tm_rgb = tonemap_photographic(hdr, K, B)
# # plt.imshow(gamma_correction(tm_rgb))
# # plt.show()
# # tm_rgb = gamma_correction(tm_rgb)
# # writeEXR("../results/hdrRenRGB.exr", tm_rgb)
# # writeHDR("../results/hdrRenRGB.hdr", tm_rgb)

# # # tonemap luminance only (xyY)
# # np.seterr(divide='ignore', invalid='ignore')
# # xyY = XYZ2xyY(lRGB2XYZ(hdr))
# # xyY[:, :, 2] = tonemap_photographic(xyY[:, :, 2], K, B)
# # tm_Y = XYZ2lRGB(xyY2XYZ(xyY))
# # tm_Y = np.clip(tm_Y, 0, None)
# # tm_Y = gamma_correction(tm_Y)
# # plt.imshow(tm_Y)
# # plt.show()
# # writeEXR("../results/hdrRenY.exr", tm_Y)
# # writeHDR("../results/hdrRenY.hdr", tm_Y)