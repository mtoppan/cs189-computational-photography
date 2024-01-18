#!/usr/bin/python

import numpy as np
from skimage import io, color
from skimage.transform import resize


def readImagesAndExposures(dir_images, nExposure, option_input, scaling=1):
    """ Read image and exposures
    """

    print('Reading images... ')
    """ WRITE YOUR CODE HERE
    """
    img_list = []
    exposure_list = []

    exposure_start = 2048
    
    if dir_images == '../data/dali_stack/':
        exposure_start = 100

    n = 0

    # loop through all images in the given image directory and read them in

    for k in range(1, nExposure + 1):
        name = dir_images+"exposure"+str(k)
        if option_input == 'rendered':
            name = name+'.jpg'
        elif option_input == 'RAW':
            name = name+'.tiff'
        im = io.imread(name)
        im = im[::scaling, ::scaling].astype(np.float64)
        exp = (1/exposure_start) * (2 ** (k-1))
        if option_input == 'rendered':
            im /= (2**(8) -1)
        elif option_input == 'RAW':
            im /= (2**(16) -1)
        img_list.append(im)
        exposure_list.append(exp)

        if n == nExposure:
            break
        n += 1

    print('Done!')

    return img_list, exposure_list

def runRadiometricCalibration(img_list, exposure_list, l, option_weight, scaling=200):
    """ Run radiometric calibration
    """

    """ WRITE YOUR CODE HERE
    """

    minZ = 0.05
    maxZ = 0.95

    log_exp = np.log(np.asarray(exposure_list))

    img_array = np.asarray(img_list)

    img_array = img_array[::, ::scaling, ::scaling, ::]

    img_array = img_array.reshape(img_array.shape[0], img_array.shape[1]*img_array.shape[2]*img_array.shape[3]).transpose()

    weights = np.linspace(0, 1, 256)

    adjust = np.where((minZ <= weights) & (weights <= maxZ))[0]
    adjusted_weights = weights[(minZ <= weights) & (weights <= maxZ)]

    w = np.zeros_like(weights)

    if option_weight == 'uniform':
        w[adjust] = 1
    elif option_weight == 'tent':
        w[adjust] = np.where(adjusted_weights < (1 - adjusted_weights), adjusted_weights, (1 - adjusted_weights))
    elif option_weight == 'Gaussian':
        w[adjust] = np.exp((-1)*(((adjusted_weights)-0.5)**2))
    elif option_weight == 'photon':
        w[adjust] = 1

    # w = np.ones(256)

    g, log_L = gsolve(img_array*(255), log_exp, l, w, option_weight)
    
    return g, w


def mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight):
    """ Merge exposure stack into HDR image
    """

    """ WRITE YOUR CODE HERE
    """

    exp_array = np.asarray(exposure_list)
    img_array = np.asarray(img_list)
    img_array = np.transpose(img_array, (1, 2, 3, 0))
    # print(exp_array.shape)
    # print(exp_array[None, None, None, :].shape)
    
    img_array *= (2**(8) - 1)
    img_array = img_array.astype(np.uint8)
    img_array = np.where(img_array > 255*0.95, 255, img_array)
    img_array = np.where(img_array < 255*0.05, 0, img_array)

    if option_input == 'rendered':
        linImg = np.exp(g[img_array])
    elif option_input == 'RAW':
        linImg = img_array

    ldrImg = img_array
    
    e = 0.001

    if option_merge == 'linear':
        if option_weight == 'photon':
            numerator = np.sum(w[ldrImg] * linImg, axis=3)
            denominator = np.sum(w[ldrImg] * exp_array[None, None, None, :] + e , axis=3)
        else:
            numerator = np.sum((w[ldrImg] * linImg / exp_array[None, None, None, :]), axis=3)
            denominator = np.sum(w[ldrImg] + e, axis=3)
        hdr = np.nan_to_num(np.divide(numerator, denominator))

    if option_merge == 'logarithmic':
        if option_weight == 'photon':
            numerator = np.sum((w[ldrImg] * exp_array[None, None, None, :] * (np.log(linImg + e) - np.log(exp_array[None, None, None, :]))), axis=3)
            denominator = np.sum(w[ldrImg] * exp_array[None, None, None, :] + e, axis=3)  
        else:
            numerator = np.sum((w[ldrImg] * (np.log(linImg + e) - np.log(exp_array[None, None, None, :]))), axis=3)
            denominator = np.sum(w[ldrImg], axis=3)
        hdr = np.exp(np.nan_to_num(np.divide(numerator, denominator)))

    return hdr

def gsolve(I, log_t, l, w, option_weight):
    """ Solve for imaging system response function

    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system response function g as well as the log film irradiance
    values for the observed pixels.

    This code is from the following paper:
    P. E. Debevec and J. Malik, Recovering High Dynamic Range Radiance Maps from Photographs, ACM SIGGRAPH, 1997

    Parameters
    ----------
    I(i, j): pixel values of pixel location number i in image j (nPixel, nExposure)
    log_t(j): log delta t, or log shutter speed for image j (nExposure)
    l: lambda, the constant that determines the amount of smoothness
    w(z): weighting function value for pixel value z (256)

    Returns
    -------
    g(z): the log exposure corresponding to pixel value z
    log_L(i) is the log film irradiance at pixel location i
    """

    n = 256
    nPixel = I.shape[0]
    nExposure = I.shape[1]

    A = np.zeros((nPixel*nExposure + n + 1, n + nPixel))
    b = np.zeros((A.shape[0],))

    # Include the data-fitting equations
    k = 0
    for i in range(nPixel):
        for j in range(nExposure):
            z = I[i,j].astype(np.uint8)

            if option_weight == 'photon':
                wij = np.exp(log_t[j])
            else:
                wij = w[z]

            A[k, z] = wij
            A[k, n+i] = -wij
            b[k] = wij * log_t[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1 # b[k] = 0
    k += 1

    # Include the smoothness equations
    for z in np.arange(1, n):
        A[k, z-1] = l*w[z]
        A[k, z] = -2*l*w[z]
        A[k, z+1] = l*w[z]
        k += 1

    print(A.shape)
    # exit()
    # Solve the system
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    g = x[:n]
    log_L = x[n:]

    return g, log_L


def tonemap_photographic(imIn, key, burn):
    """ Implementation of Reinhard et al., Photographic Tone Reproduction for
    Digital Images, SIGGRAPH 2002.
    """
    
    """ WRITE YOUR CODE HERE
    """
    e = 0.001

    im_m = np.exp((1 / (imIn.shape[0] * imIn.shape[1])) * np.sum(np.log(imIn + e)))
    im_tilda = (key / im_m) * imIn
    im_white = burn * np.max(im_tilda)

    imOut = (im_tilda * (1 + (im_tilda/(im_white**2)))) / (1 + im_tilda)
    
    return imOut

def gamma_correction(img_in):
    """ WRITE YOUR CODE HERE
    """
    mult = 0.25 / np.mean(color.rgb2gray(img_in))
    brighter = np.clip((img_in* mult), 0, 1)

    img_out = np.where(brighter <= 0.0031308, (12.92*brighter), (((1 + 0.055) * (brighter**(1/(2.4)))) - 0.055))

    return img_out

def XYZ2xyY(XYZ):
    X = XYZ[:,:,0]
    Y = XYZ[:,:,1]
    Z = XYZ[:,:,2]
    
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    
    xyY = np.dstack((x, y, Y))
    return xyY

def xyY2XYZ(xyY):
    x = xyY[:,:,0]
    y = xyY[:,:,1]
    Y = xyY[:,:,2]
    
    sum_XYZ = Y/y
    X = x*sum_XYZ
    Z = sum_XYZ - X - Y
    XYZ = np.dstack((X, Y, Z))
    return XYZ
    
