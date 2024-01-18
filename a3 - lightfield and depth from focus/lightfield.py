#!/usr/bin/python

import numpy as np
from skimage import io, color, filters
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from cp_hw2 import lRGB2XYZ
import cv2

BLOCK = 16

##################### INITIALS ####################
# Load the lightfield image in Python, and create #
# from it a 5-dimensional array L(u,v,s,t,c).     #
#                                                 #
# INPUT:                                          #
#     - imName = the lightfield image             #
# OUTPUT:                                         #
#     - L = 5D array consisting of u, v, s, t, c  #
###################################################
def initials(imName):
    im = io.imread(imName)

    h = im.shape[0] 
    w = im.shape[1]
    
    sShape = w // BLOCK
    tShape = h // BLOCK

    L = np.transpose((im.reshape(BLOCK, tShape, BLOCK, sShape, 3, order='F')), (2, 0, 1, 3, 4)).astype(np.uint8)

    return L

######################################################################

###################### MOSAIC #####################
# Take slices of the lightfield of the form       #
# L(u = u0, v = v0, s, t, c)  and present them in #
# a mosaic of 16 x 16.                            #
#                                                 #
# INPUT:                                          #
#     - l: the reshaped lightfield image          #
# OUTPUT:                                         #
#     - mosaic: a recomposed image of 16x16 pics  #
###################################################
def subApertureViews(L):
    s = L.shape[2]
    t = L.shape[3]

    L = np.transpose(L, (1, 0, 2, 3, 4))
    mosaic = np.transpose(L, (0, 2, 1, 3, 4)).reshape(BLOCK*s, BLOCK*t, 3)

    return mosaic

######################################################################

############ BONUS: CIRCULAR APERTURE #############
# Helper to constrain DOF based on a circular     #
# aperture.                                       #
#                                                 #
# INPUT:                                          #
#     - u: the u value at which we get the image  #
#     - v: the v at which we get the image        #
#     - aperture: the size of the aperture        #
# OUTPUT:                                         #
#     - T/F: True if in range of A; else, False   #
###################################################
def circularAperture(u, v, aperture):
    if ((u**2 + v**2)**(1/2)) <= (aperture // 2):
        return True
    return False

############# BONUS: SQUARE APERTURE ##############
# Helper to constrain DOF based on a square       #
# aperture.                                       #
#                                                 #
# INPUT:                                          #
#     - u: the u value at which we get the image  #
#     - v: the v at which we get the image        #
#     - aperture: the size of the aperture        #
# OUTPUT:                                         #
#     - T/F: True if in range of A; else, False   #
###################################################
def squareAperture(u, v, aperture):
    if (abs(u) <= (aperture / 2)) and (abs(v) <= (aperture / 2)):
        return True
    return False

############### BONUS: MAX APERTURE ###############
# Helper to constrain DOF range based on largest  #
# aperture.                                       #
#                                                 #
# INPUT:                                          #
#     - u: the u value at which we get the image  #
#     - v: the v at which we get the image        #
#     - maxUV: the size of the largest aperture   #
# OUTPUT:                                         #
#     - T/F: True if in range of A; else, False   #
###################################################
def maxAperture(u, v, maxUV):
    if abs(u) <= maxUV and abs(v) <= maxUV:
        return True
    return False

################ HELPER: INTERP ###################
# Helper to interpolate s and t over a color dim. #
#                                                 #
# INPUT:                                          #
#     - l: image at u, v for a color channel      #
#     - u: the u at which we get the image        #
#     - v: the u at which we get the image        #
#     - d: the depth at which to simulate         #
# OUTPUT:                                         #
#     - int: interpolated image for the channel   #
###################################################
def interp(im, u, v, d):
    s = np.arange(im.shape[1])
    t = np.arange(im.shape[0])
    i = interp2d(s, t, im)
    int = i(s+(d*u), t-(d*v))
    return int

#################### REFOCUS ######################
# Refocus an image to different apertures with    #
# the given lightfield.                           #
#                                                 #
# INPUT:                                          #
#     - l: the reshaped lightfield image          #
#     - d: the depth at which to simulate         #
# OUTPUT:                                         #
#     - refocus: the refocused image with 'DOF'   #
###################################################
def refocus(l, d, aperture='max', a=1):
    maxUV = (BLOCK - 1) / 2
    uRange = np.arange(BLOCK) - maxUV
    vRange = np.arange(BLOCK) - maxUV

    refocus = np.zeros((l.shape[2], l.shape[3], l.shape[4]))

    for u in range(len(uRange)):
        for v in range(len(vRange)):
            #APERTURE STUFF: TAKE IN AN APERTURE, EQUATION FOR A, CHECK THAT VALUES IN URANGE AT INDICES OF U, V ARE SOLID (IF STATEMENT)
            if (aperture == 'max' and maxAperture(uRange[u], vRange[v], maxUV) == False) or (aperture == 'circle' and circularAperture(uRange[u], vRange[v], a) == False) or (aperture == 'square' and squareAperture(uRange[u], vRange[v], a) == False):
                continue

            imr = l[u, v, :, :, 0]
            img = l[u, v, :, :, 1]
            imb = l[u, v, :, :, 2]
            ir = interp(imr, uRange[u], vRange[v], d)
            ig = interp(img, uRange[u], vRange[v], d)
            ib = interp(imb, uRange[u], vRange[v], d)
            i = np.dstack((ir, ig, ib))

            refocus += i


    refocus = (refocus / (BLOCK ** 2)) / 255

    return refocus

################### SEE REFOCUS ###################
# Prints the refocused image at depths of 0.5, 0, #
# -0.5, -1, -1.5, and -2.                         #
#                                                 #
# INPUT:                                          #
#     - l: the reshaped lightfield image          #
# OUTPUT:                                         #
#     - none                                      #
###################################################
def seeRefocus(l):
    i2 = refocus(l, -2)
    i15 = refocus(l, -1.5)
    i1 = refocus(l, -1)
    i05 = refocus(l, -0.5)
    i0 = refocus(l, 0)
    mini05 = refocus(l, 0.5)

    showDepths = plt.figure()
    showDepths.add_subplot(2,3,1)
    plt.title('Depth=0.5')
    plt.imshow(mini05)
    showDepths.add_subplot(2,3,2)
    plt.title('Depth=0')
    plt.imshow(i0)
    showDepths.add_subplot(2,3,3)
    plt.title('Depth=-0.5')
    plt.imshow(i05)
    showDepths.add_subplot(2,3,4)
    plt.title('Depth=-1')
    plt.imshow(i1)
    showDepths.add_subplot(2,3,5)
    plt.title('Depth=-1.5')
    plt.imshow(i15)
    showDepths.add_subplot(2,3,6)
    plt.title('Depth=-2')
    plt.imshow(i2)
    plt.show()
    # plt.imsave('./res/FocalStack.png', showDepths)

############### HELPER: FOCALSTACK ################
# Create a focal stack of a given lightmap image  #
# at different depths.                            #
#                                                 #
# INPUT:                                          #
#     - l: the reshaped lightfield image          #
#     - ds: the depths at which to simulate       #
# OUTPUT:                                         #
#     - stack: the focal stack                    #
###################################################
def focalStack(l, ds, aperture='max', aCount=1):
    stack = np.zeros((l.shape[2], l.shape[3], l.shape[4], len(ds)))

    for d in range(len(ds)):
        for a in np.linspace(1, 16, aCount):
            stack[:, :, :, d] = refocus(l, ds[d], aperture, a)
    
    # print(stack.shape)

    return stack

######################################################################

############# HELPER: GAMMA DECODING ##############
# Given a gamma-encoded image, reverse the gamma- #
# encoding process to restore the original image. #
#                                                 #
# INPUT:                                          #
#     - imIn: the gamma-encoded image             #
# OUTPUT:                                         #
#     - imOut: the non-gamma-encoded image        #
###################################################
def gammaDecoding(imIn):
    imOut = np.where(imIn <= 0.0404482, imIn/12.92, ((imIn + 0.055) / 1.055)**(2.4))
    return imOut

################## ALL IN FOCUS ###################
# Using a focal stack, calculate weights to       #
# reshape the images of different depths into one #
# all-in-focus image.                             #
#                                                 #
# INPUT:                                          #
#     - fs: the focal stack                       #
#     - k1: kernel to blur the low-frequency      #
#           component.                            #
#     - k2: kernel to blur the high-frequency     #
#           component.                            #
#     - s1: sigma for blurring the low-frequency  #
#           component.                            #
#     - s2: sigma for blurring the high-frequency #
#           component.                            #
#     - ds: the depths present in the focal stack #
# OUTPUT:                                         #
#     - focused: the all-in-focus image           #
#     - depth: the depth image based on weights   #
###################################################
def allInFocus(fs, k1, k2, s1, s2, ds):
    # convert each image in stack to XYZ colorspace
    xyz = np.zeros((fs.shape))
    for i in range(len(ds)):
        xyz[:, :, :, i] = lRGB2XYZ(gammaDecoding(fs[:, :, :, i]))
        
    # extract luminance channel
    luminance = xyz[:, :, 1, :]

    # blur with gaussian kernel of sd sigma
    gaus = cv2.GaussianBlur(luminance, [k1, k1], s1)

    #compute high frequency component by subtracting blurry image from the original
    hf = luminance - gaus

    #compute sharpness weight by blurring the SQUARE of high-frequency with another Gaussian kernel of sd sig2
    w = cv2.GaussianBlur(hf**2, [k2, k2], s2)

    w = np.stack((w, w, w), axis = 2)

    ds = np.asarray(ds + abs(np.min(ds)))
    ds = ds / np.max(ds)

    focusedNumerator = np.sum(w * fs, axis=-1)
    focusedDenominator = np.sum(w, axis=-1)

    depthNumerator = np.sum(w * ds[None, None, None, :], axis=-1)
    depthDenominator = np.sum(w, axis=-1)

    focused = np.nan_to_num(np.divide(focusedNumerator, focusedDenominator))
    depth = np.nan_to_num(np.divide(depthNumerator, depthDenominator))

    return focused, depth

######################################################################

#################### MAIN: RUN ####################
# Put the functions above into practice to fully  #
# analyze and run through an image.               #
#                                                 #
# INPUT:                                          #
#     - None.                                     #
# OUTPUT:                                         #
#     - None.                                     #
###################################################
def run():
    L = initials('./data/chessboard_lightfield.png')

    mosaic = subApertureViews(L)
    plt.imshow(mosaic)
    plt.show()

    seeRefocus(L)

    ds = np.linspace(-1.5, 0, 10)
    stack = focalStack(L, ds, 'max', 1)

    f, d = allInFocus(stack, 5, 33, 2, 8, ds)
    plt.imshow(f)
    plt.show()
    plt.imshow(d)
    plt.show()


run()