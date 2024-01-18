import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline
import pathlib


def pInitials(im):
    shape = np.shape(im)
    depth = im.dtype

    print("The width of the image is", shape[0])
    print("The height of the image is", shape[1])
    print("The bit depth of the image is", depth)

    # convert the image into a double-precision array
    return im.astype(np.float64)

def linearization(black, white, imArray):
    #linear transform to the image so that black is mapped to zero and white is mapped to one
    # so scale so that the black value is one apart from the white value
    # then shift values so that black is at zero and white is at one
    # then if the value is less than zero, replace with zero
    # if the values is greater than one, replace with one
    scale = (white - black)
    blackShift = black / scale
    
    rescaledIMArray = imArray / scale
    shiftedIMArray = rescaledIMArray - blackShift

    blackIMArray = np.where(shiftedIMArray < 0, 0, shiftedIMArray)

    fullIMArray = np.where(blackIMArray > 1, 1, blackIMArray)

    return fullIMArray

def findBayerPattern(imArray):
    #this can be done with dcraw— dcraw -i -v data/Thayer.cr2
    #but also i have a feeling that this is NOT what is wanted
    # could also do explained trial and error tbh
    #GRBG
    g = imArray[0::2, 0::2]
    r = imArray[0::2, 1::2]
    b = imArray[1::2, 0::2]
    im_grbg = np.dstack((r, g, b))

    #RGGB
    r = imArray[0::2, 0::2]
    g1 = imArray[0::2, 1::2]
    g2 = imArray[1::2, 0::2]
    b = imArray[1::2, 1::2]
    im_rggb= np.dstack((r, (g1+g2)/2, b))

    #BGGR
    b = imArray[0::2, 0::2]
    g = imArray[0::2, 1::2]
    r = imArray[1::2, 1::2]
    im_bggr = np.dstack((r, g, b))

    #GRBG
    b = imArray[0::2, 0::2]
    g = imArray[0::2, 1::2]
    r = imArray[1::2, 0::2]
    im_gbrg = np.dstack((r, g, b))
    return im_grbg, im_rggb, im_bggr, im_gbrg

def grayWorldWB(imArray):
# • Compute per-channel average.
    r_avg = np.mean(imArray[:, :, 0])
    g_avg = np.mean(imArray[:, :, 1])
    b_avg = np.mean(imArray[:, :, 2])
# • Normalize each channel by its average, then by the green average.
    r_norm = (imArray[:, :, 0] * g_avg) / r_avg
    g_norm = imArray[:, :, 1]
    b_norm = (imArray[:, :, 2] * g_avg) / b_avg
    
    imArray_WB = np.dstack((r_norm, g_norm, b_norm))
    
    return imArray_WB

def whiteWorldWB(imArray):
# • Compute per-channel max.
    r_max = (imArray[:, :, 0]).max()
    g_max = (imArray[:, :, 1]).max()
    b_max = (imArray[:, :, 2]).max()
# • Normalize each channel by its max, then by the green max.
    r_norm = (imArray[:, :, 0] * g_max) / r_max
    g_norm = imArray[:, :, 1]
    b_norm = (imArray[:, :, 2] * g_max) / b_max
    
    imArray_WB = np.dstack((r_norm, g_norm, b_norm))
    
    return imArray_WB

def dcrawWB(imArray, r, g, b):
    r_norm = (imArray[:, :, 0] * g) / r
    g_norm = imArray[:, :, 1]
    b_norm = (imArray[:, :, 2] * g) / b

    imArray_WB = np.dstack((r_norm, g_norm, b_norm))

    return imArray_WB

def manualWB(imArray):
    io.imshow(imArray)
    white_point = np.asarray(plt.ginput(1))[0]
    y = int(white_point[0])
    x = int(white_point[1])
    r = imArray[x][y][0]
    g = imArray[x][y][1]
    b = imArray[x][y][1]
    
    r_scaled = imArray[:,:,0] / r
    g_scaled = imArray[:,:,1] / g
    b_scaled = imArray[:,:,2] / b

    imArray_WB = np.dstack((r_scaled, g_scaled, b_scaled))

    return imArray_WB

def demosaicing(imArray, w, h, bayered=0):

    rx = np.arange(0, w, 2)
    ry = np.arange(0, h, 2)

    g1x = np.arange(1, w, 2)
    g1y =  np.arange(0, h, 2)

    g2x = np.arange(0, w, 2)
    g2y = np.arange(1, h, 2)

    bx = np.arange(1, w, 2)
    by = np.arange(1, h, 2)

    #grid them with np.meshgrid
    r_x, r_y = np.meshgrid(rx, ry)
    g1_x, g1_y = np.meshgrid(g1x, g1y)
    g2_x, g2_y = np.meshgrid(g2x, g2y)
    b_x, b_y = np.meshgrid(bx, by)
    
    # #bilinear interp them on meshgrid
    r_i = imArray[r_y, r_x]
    g1_i = imArray[g1_y, g1_x]
    g2_i = imArray[g2_y, g2_x]
    b_i = imArray[b_y, b_x]

    print(rx.shape)
    print(r_y.shape)
    
    if bayered == 0 : #if it's the original image
        r_interp = interp2d(rx, ry, r_i)
        g1_interp = interp2d(g1x, g1y, g1_i)
        g2_interp = interp2d(g2x, g2y, g2_i)
        b_interp = interp2d(bx, by, b_i)
    
    else : #if it's already gone thru whitebalancing
        r_interp = interp2d(rx, ry, r_i[:,:,0])
        g1_interp = interp2d(g1x, g1y, g1_i[:,:,1])
        g2_interp = interp2d(g2x, g2y, g2_i[:,:,1])
        b_interp = interp2d(bx, by, b_i[:,:,2])

    r = r_interp(rx, ry)

    g1 = g1_interp(g1x, g1y)
    g2 = g2_interp(g2x, g2y)
    g = (g1 + g2) / 2

    b = b_interp(bx, by)

    imArray_dm = np.dstack((r, g, b))
    
    return imArray_dm

def colorSpaceCorrection(imArray):
    M_rgb_xyz = np.array([[0.4124564, 0.3575761, 0.1804375], 
                          [0.2126729, 0.7151522, 0.0721750], 
                          [0.0193339, 0.1191920, 0.9503041]])
    M_xyz_cam = np.array([[8532, -701, -1167], 
                          [-4095, 11879, 2508], 
                          [-797, 2424, 7010]])
    
    M_xyz_cam = (M_xyz_cam) / 10000

    M_rgb_cam = np.matmul(M_rgb_xyz, M_xyz_cam)

    M_rgb_cam = (M_rgb_cam) / (np.sum(M_rgb_cam, axis=1))
    
    imArray_c = np.zeros(imArray.shape)

    imT = imArray.transpose(2,0,1)
    imR = imT.reshape(3,-1)
    imC = np.matmul((np.linalg.inv(M_rgb_cam)),imR)
    imC = imC.reshape(3,imT.shape[1], imT.shape[2]) 
    imArray_c = imC.transpose(1,2,0)
    print(M_rgb_cam)

    return imArray_c

def brightnessAndGammaCorrection(imArray):
    mult = 0.25 / np.mean(color.rgb2gray(imArray))
    brighter = np.clip((imArray * mult), 0, 1)

    imArray_bg = np.where(brighter <= 0.0031308, (12.92*brighter), (((1 + 0.055) * (brighter**(1/(2.4)))) - 0.055))

    return imArray_bg

def compression(imArray):
    io.imsave('./results/nocompression.png', imArray)
    io.imsave('./results/compression95.jpg', imArray, quality=95)
    io.imsave('./results/compression75.jpg', imArray, quality=75)
    io.imsave('./results/compression55.jpg', imArray, quality=55)
    io.imsave('./results/compression35.jpg', imArray, quality=35)
    io.imsave('./results/compression25.jpg', imArray, quality=25)
    io.imsave('./results/compression15.jpg', imArray, quality=15)
    io.imsave('./results/compression10.jpg', imArray, quality=10)
    io.imsave('./results/compression5.jpg', imArray, quality=5)
    png_path = pathlib.Path('./results/nocompression.png')
    jpg_path = pathlib.Path('./results/compression95.jpg')

    size_ratio = png_path.stat().st_size / jpg_path.stat().st_size

    return size_ratio
    


def main():
    im = io.imread('data/Thayer.tiff')
    imArray = pInitials(im)
    # io.imshow(imArray)
    # plt.savefig('./results/initials.jpg')
    # plt.show()

    imArray = linearization(2044, 16383, imArray)
    im_OG = imArray
    w = im_OG.shape[1]
    h = im_OG.shape[0]
    # io.imshow(imArray)
    # plt.savefig('./results/linearization.jpg')
    # plt.show()

    im_grbg, im_rggb, im_bggr, im_gbrg = findBayerPattern(imArray)
    # bayer_fig = plt.figure()
    # bayer_fig.add_subplot(2, 2, 1)
    # plt.title('GRBG')
    # plt.imshow(np.clip(im_grbg*3, 0, 1))
    # bayer_fig.add_subplot(2, 2, 2)
    # plt.title('RGGB')
    # plt.imshow(np.clip(im_rggb*3, 0, 1))
    # bayer_fig.add_subplot(2, 2, 3)
    # plt.title('BGGR')
    # plt.imshow(np.clip(im_bggr*3, 0, 1))
    # bayer_fig.add_subplot(2, 2, 4)
    # plt.imshow(np.clip(im_gbrg*3, 0, 1))
    # plt.title('GBRG')
    # plt.savefig('./results/bayers.jpg')
    # plt.show()

    # io.imshow(im_rggb)
    # plt.savefig('./results/rggb.jpg')
    # plt.show()

    im_gw = grayWorldWB(im_rggb)
    im_ww = whiteWorldWB(im_rggb)
    im_dcraw = dcrawWB(im_rggb, 2.165039, 1.0, 1.643555)
    im_manual = manualWB(im_rggb)

    io.imshow(im_manual)
    plt.savefig('./results/manual1.jpg')
    plt.show()

    im_manual = manualWB(im_rggb)

    io.imshow(im_manual)
    plt.savefig('./results/manual2.jpg')
    plt.show()

    im_manual = manualWB(im_rggb)

    io.imshow(im_manual)
    plt.savefig('./results/manual3.jpg')
    plt.show()

    im_manual = manualWB(im_rggb)


    wb_fig = plt.figure()
    wb_fig.add_subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(im_rggb)
    wb_fig.add_subplot(2, 2, 2)
    plt.title('Gray world')
    plt.imshow(im_gw)
    wb_fig.add_subplot(2, 2, 3)
    plt.title('White world')
    plt.imshow(im_ww)
    wb_fig.add_subplot(2, 2, 4)
    plt.imshow(im_dcraw)
    plt.title('dcraw')
    plt.savefig('./results/whitebalancing.jpg')
    plt.show()
    io.imshow(im_gw)
    plt.savefig('./results/grayworld.jpg')
    plt.show()

    im_dm = demosaicing(im_OG, w, h)
    io.imshow(im_dm)
    plt.savefig('./results/demosaiced.jpg')
    plt.show()
    im_dm = grayWorldWB(im_dm)

    w_wb = im_gw.shape[1]
    h_wb = im_gw.shape[0]
    im_dmp = demosaicing(im_gw, w_wb, h_wb, 1)
    io.imshow(im_dmp)
    plt.savefig('./results/demosaicedPostWB.jpg')
    plt.show()

    im_corrected = colorSpaceCorrection(im_dm)
    io.imshow(im_corrected)
    plt.savefig('./results/colorcorrected.jpg')
    plt.show()
    im_gb = brightnessAndGammaCorrection(im_corrected)
    io.imshow(im_gb)
    plt.savefig('./results/gammacorrected.jpg')
    plt.show()

    wb_fig2 = plt.figure()
    wb_fig2.add_subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(im_gb)
    wb_fig2.add_subplot(2, 2, 2)
    plt.title('Gray world')
    plt.imshow(grayWorldWB(im_gb))
    wb_fig2.add_subplot(2, 2, 3)
    plt.title('White world')
    plt.imshow(whiteWorldWB(im_gb))
    wb_fig2.add_subplot(2, 2, 4)
    plt.title('dcraw')
    plt.imshow(dcrawWB(im_gb, 2.165039, 1.0, 1.643555))
    plt.savefig('./results/whitebalancingpost.jpg')
    plt.show()

    size_ratio = compression(im_gb)
    print(size_ratio)



main()