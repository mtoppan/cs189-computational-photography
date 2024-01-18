import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from cp_hw2 import lRGB2XYZ
from skimage import io, color
from cv2 import GaussianBlur
from cp_hw4 import integrate_frankot, integrate_poisson, load_sources
import math

################### INITIALS ######################
# Read in images and flatten them into an array   #
# of pixel values.                                #
#                                                 #
# INPUT:                                          #
#     - pathName: the prefix and path for each    #
#           image                                 #
#     - nFiles: the number of images to stack     #
#     - q: the question at hand, to adjust the    #
#           suffix (q=1 for Q1, q=2 for Q2)       #
# OUTPUT:                                         #
#     - I: the image stack                        #
#     - h: the height of each original image      #
#     - w: the width of each original image       #
###################################################
def initials(pathName, nFiles, q=1):
    print("Reading images...")
    I = []

    for i in range(1, nFiles+1):
        if q == 1:
            filePath = pathName + str(i) + '.tif'
        elif q == 2:
            filePath = pathName + str(i) + '.tiff'
        image = io.imread(filePath).astype(np.float64)
        image /= 2**(16) - 1
        if image.shape[0] > 512 or image.shape[1] > 512:
            print("Image" + str(i) + " cropped.")
            if image.shape[0] > image.shape[1]:
                d = math.ceil(image.shape[0] / 512)
            else:
                d = math.ceil(image.shape[1] / 512)
            image = image[::d, ::d]
        h = image.shape[0]
        w = image.shape[1]

        l = np.ravel(lRGB2XYZ(image)[:, :, 1])

        I.append(l)
    
    I = np.asarray(I)

    return I, h, w

################## GET NORMALS ####################
# Use SVD to calculate the pseudonormals, and     #
# thereby the normal and albedo map, of the       #
# stereo image.                                   #
#                                                 #
# INPUT:                                          #
#     - I: the stack of images                    #
# OUTPUT:                                         #
#     - b: the pseudonormal matrix                #
#     - l: the light matrix                       #
#     - a: the albedo based on the pseudonormals  #
#     - n: the normals based on the pseudonormals #
###################################################
def getNormals(I):
    print("Getting the normal and albedo...")
    u, s, v = np.linalg.svd(I, full_matrices=False)

    s = np.diag(s)
    s = np.sqrt(s)

    b = np.matmul(s, v)
    b = b[:3]
    l = np.matmul(s, u)
    l = l[:3]

    a = np.linalg.norm(b, axis=0)
    n = b / (a)

    return b, l, a, n

####### VISUALIZE: SHOW ALBEDO AND NORMALS ########
# Display the albedos and normals.                #
#                                                 #
# INPUT:                                          #
#     - a: the albedo based on the pseudonormals  #
#     - n: the normals based on the pseudonormals #
#     - h: the height of each original image      #
#     - w: the width of each original image       #
# OUTPUT:                                         #
#     - Nothing returned                          #
###################################################
def showAlbedoAndNorms(a, n, h, w):
    print("Showing the normal and albedo...")
    n = np.transpose(n)

    n = np.reshape(n, (h, w, 3))
    a = np.reshape(a, (h, w))

    n = (n - (np.min(n)))
    n = n / np.max(n)

    plt.imshow(n)
    plt.show()

    plt.imshow(a, cmap='gray')
    plt.show()

###### VISUALIZE: SHOW ALBEDOQ AND NORMALSQ #######
# Display the albedos and normals from B x Q.     #
#                                                 #
# INPUT:                                          #
#     - b: the pseudonormal matrix                #
#     - h: the height of each original image      #
#     - w: the width of each original image       #
# OUTPUT:                                         #
#     - Nothing returned                          #
###################################################
def showAlbedoAndNormsQ(b, h, w):
    print("Showing the normal and albedo...")
    q = np.array([[8, 9, 7], [1, 2, 3], [9,  2,  1]])
    q = np.transpose(q)
    q = np.linalg.inv(q)

    bq = np.matmul(q, b)

    a = np.linalg.norm(bq, axis=0)
    n = bq / (a + 0.000001)

    n = np.transpose(n)

    n = np.reshape(n, (h, w, 3))
    a = np.reshape(a, (h, w))

    n = (n - (np.min(n)))
    n = n / np.max(n)

    plt.imshow(n)
    plt.show()

    plt.imshow(a, cmap='gray')
    plt.show()

############# ENFORCE INTEGRABILITY ###############
# Calculate a delta by which to multiply the      #
# pseudonormals s.t. they are integrable.         #
#                                                 #
# INPUT:                                          #
#     - b: the pseudonormal matrix                #
#     - h: the height of each original image      #
#     - w: the width of each original image       #
#     - k: the kernel size for Gaussian blur      #
#     - s: the sigma size for Gaussian blur       #
#     - g: the gbr matrix to adjust b             #
# OUTPUT:                                         #
#     - bDelta: the integrable pseudonormal       #
#           matrix                                #
#     - aDelta: the albedo based on the           #
#           integrable pseudonormals              #
#     - nDelta: the normals based on the          #
#           integrable pseudonormals              #
###################################################
def enforceIntegrability(b, h, w, k, s, g):
    print("Enforcing integrability...")

    be = np.transpose(b)
    be = np.reshape(be, (h, w, 3))
    be0 = be[:, :, 0]
    be1 = be[:, :, 1]
    be2 = be[:, :, 2]

    bg = np.zeros((h, w, 3))
    
    for c in range(3):
        bg[:, :, c] = GaussianBlur(be[:, :, c], (k, k), s)

    by0, bx0 = np.gradient(bg[:, :, 0], edge_order=2)
    by1, bx1 = np.gradient(bg[:, :, 1], edge_order=2)
    by2, bx2= np.gradient(bg[:, :, 2], edge_order=2)

    A1 = np.asarray(be0 * bx1 - be1 * bx0)
    A2 = np.asarray(be0 * bx2 - be2 * bx0) 
    A3 = np.asarray(be1 * bx2 - be2 * bx1) 
    A4 = np.asarray(-be0 * by1 + be1 * by0)
    A5 = np.asarray(-be0 * by2 + be2 * by0)
    A6 = np.asarray(-be1 * by2 + be2 * by1)
    
    A = np.hstack((A1.reshape(-1, 1), A2.reshape(-1, 1), A3.reshape(-1, 1), A4.reshape(-1, 1), A5.reshape(-1, 1), A6.reshape(-1, 1)))

    u, s, v= np.linalg.svd(A, full_matrices=False)
    x = v[-1, :]

    delta = np.asarray([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])

    bDelta = np.matmul(np.linalg.inv(np.transpose(g)), np.matmul(np.linalg.inv(delta), b))
    aDelta = np.linalg.norm(bDelta, axis=0, keepdims=True)
    nDelta = bDelta / (aDelta + 0.000001)

    return bDelta, aDelta, nDelta

############## NORMAL INTEGRATION #################
# Calculate the surface based on the normal map.  #
#                                                 #
# INPUT:                                          #
#     - n: image at i, j for a color channel      #
#     - h: the x at which we get the image        #
#     - w: the y at which we get the image        #
#     - e: small val to add to division           #
# OUTPUT:                                         #
#     - sPoisson: the surface calculated with     #
#           Poisson integration                   #
#     - sFrankot: the surface calculated with     #
#           Frankot integration                   #
###################################################
def normalIntegration(n, h, w, e):
    print("Integrating the normals...")
    
    n = np.transpose(n)
    n = np.reshape(n, (h, w, 3))

    dx = n[:, :, 0] / (n[:, :, 2] + e)
    dy = n[:, :, 1] / (n[:, :, 2] + e)

    sPoisson = integrate_poisson(dx, dy)
    sPoisson = sPoisson - np.min(sPoisson)
    sPoisson = (sPoisson / np.max(sPoisson))

    sFrankot = integrate_frankot(dx, dy)
    sFrankot = sFrankot - np.min(sFrankot)
    sFrankot = (sFrankot / np.max(sFrankot))
    
    return sPoisson, sFrankot

########## VISUALIZE: SHOW DEPTH AND 3D ###########
# Display the depth and surface map.              #
#                                                 #
# INPUT:                                          #
#     - s: the calculated surface                 #
#     - h: the height of each original image      #
#     - w: the width of each original image       #
# OUTPUT:                                         #
#     - Nothing returned                          #
###################################################
def showDepthAnd3D(s, h, w):
    print("Showing the depth and surface...")

    plt.imshow(-s, cmap='binary')
    plt.show()

    x, y = np.meshgrid(np.arange(h), np.arange(w))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ls = LightSource()
    color_shade = ls.shade(-s, plt.cm.gray)
    ax.plot_surface(np.transpose(x), np.transpose(y), -s, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis('off')
    plt.show()

########## CALIBRATED PHOTOMETRIC STEREO ##########
# Calculate the albedos and normals given light   # 
# data.                                           #
#                                                 #
# INPUT:                                          #
#     - I: the image stack                        #
#     - g: the gbr matrix to adjust b             #
# OUTPUT:                                         #
#     - b: the pseudonormal matrix                #
#     - a: the albedo based on the pseudonormals  #
#     - n: the normals based on the pseudonormals #
###################################################
def calibratedPhotometricStereo(I, g):
    print("Calibrating with given lights...")
    sources = load_sources()
    
    sourcesInv = np.linalg.pinv(sources)

    b = np.matmul(sourcesInv, I)
    b = np.matmul(np.linalg.inv(np.transpose(g)), b)

    a = np.linalg.norm(b, axis=0)
    n = b / (a + 0.00001)

    return b, a, n

###################### RUN ########################
# Execute all functions in intended order for     #
# desired functionality.                          #
#                                                 #
# INPUT:                                          #
#     - Nothing taken                             #
# OUTPUT:                                         #
#     - Nothing returned                          #
###################################################
def run():
    ####Q1####
    ##UNCALIBRATED##
    print("##########UNCALIBRATED##########")
    I, h, w = initials('./data/input_', 7) 

    g1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    b, l, a, n = getNormals(I)

    showAlbedoAndNorms(a, n, h, w)

    showAlbedoAndNormsQ(b, h, w)

    bDelta, aDelta, nDelta = enforceIntegrability(b, h, w, 41, 10, g1)

    showAlbedoAndNorms(aDelta, nDelta, h, w)

    sPoisson, sFrankot = normalIntegration(nDelta, h, w, 0.0000001)

    showDepthAnd3D(sFrankot, h, w)

    ##CALIBRATED##
    print("##########CALIBRATED##########")
    g2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bC, aC, nC = calibratedPhotometricStereo(I, g2)

    showAlbedoAndNorms(aC, nC, h, w)

    sPoisson, sFrankot = normalIntegration(nC, h, w, 0.0000001)

    showDepthAnd3D(-sFrankot, h, w)

    ####Q2####
    ##BUTTERFLY (not shiny)##
    print("##########BUTTERFLY##########")
    I, h, w = initials('./butterflyData/butterfly_', 7, q=2) 

    b, l, a, n = getNormals(I)

    showAlbedoAndNorms(a, n, h, w)

    bDelta, aDelta, nDelta = enforceIntegrability(b, h, w, 41, 10, g1)

    showAlbedoAndNorms(aDelta, nDelta, h, w)
    
    nDelta = (nDelta - (np.min(nDelta)))
    nDelta = nDelta / np.max(nDelta)

    sPoisson, sFrankot = normalIntegration(nDelta, h, w, 0.00000001)

    showDepthAnd3D(sPoisson, h, w)
    
    ##DEER (shiny)##
    print("##########DEER##########")
    I, h, w = initials('./deerData/deer_', 7, q=2) 

    b, l, a, n = getNormals(I)

    showAlbedoAndNorms(a, n, h, w)

    bDelta, aDelta, nDelta = enforceIntegrability(b, h, w, 41, 10, g1)

    showAlbedoAndNorms(aDelta, nDelta, h, w)
    
    nDelta = (nDelta - (np.min(nDelta)))
    nDelta = nDelta / np.max(nDelta)

    sPoisson, sFrankot = normalIntegration(nDelta, h, w, 0.00000001)

    showDepthAnd3D(sFrankot, h, w)

run()
