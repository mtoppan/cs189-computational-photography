import numpy as np
from loadVid import loadVid
import matplotlib.pyplot as plt
from cp_hw2 import lRGB2XYZ
from scipy.signal import correlate2d
from scipy.interpolate import interp2d

################### READ VIDEO ####################
# Read in and parse a video into a sequence of    #
# images.                                         #
#                                                 #
# INPUT:                                          #
#     - vidName: the path to a video to load in   #
# OUTPUT:                                         #
#     - sample: every third frame of the video,   #
#               in an array                       #
###################################################
def readVid(vidName):
    vid = loadVid(vidName)

    sample = vid[::3]

    print("Video imported")

    return sample

########## NORMALIZED CROSS CORRELATION ###########
# Calculate the normalized cross correlation on   #
# an image in order to match a template patch     #
# within a region of a given image.               #
#                                                 #
# INPUT:                                          #
#     - frame: a video frame (image)              #
#     - template: the middlemost image of the vid #
#     - xt: the leftmost coord of the template    #
#     - yt: the topmost coord of the template     #
# OUTPUT:                                         #
#     - sx, sy: the value by which to shift the   #
#           image to match the template           #
###################################################
def normalizedCrossCorrelation(frame, template, xt, yt):
    print('nxc-ing')

    #get luminance channel
    template = lRGB2XYZ(template)[:, :, 1]
    frame = lRGB2XYZ(frame)[:, :, 1]

    #set up range for template image
    g = template[xt:(xt+50), yt:(yt+50)]
    gbar = np.mean(g)

    #get range of image to compare to patch and get averaged range
    I = frame[(xt-100):(xt+150), (yt-100):(yt+150)]
    Ibar = np.mean(I)

    #correlate2d over normalize
    num = correlate2d(I-Ibar, g-gbar, mode='same')
    denom = np.sqrt(np.sum((g-gbar)**2) * np.sum((I-Ibar)**2))

    h = num / denom

    sx, sy = np.unravel_index(np.argmax(h), h.shape)

    return sx, sy

################ HELPER: INTERP ###################
# Helper to shift an image to focus at a patch.   #
#                                                 #
# INPUT:                                          #
#     - l: image at i, j for a color channel      #
#     - x: the x at which we get the image        #
#     - y: the y at which we get the image        #
# OUTPUT:                                         #
#     - int: interpolated image for the channel   #
###################################################
def interp(im, x, y):
    w = np.arange(im.shape[1])
    h = np.arange(im.shape[0])
    i = interp2d(w, h, im)
    int = i(w+(x), h+(y))
    return int

############## REFOCUS UNSTRUCTURED ###############
# Given a video, adjust each frame and composite  #
# them all into an image to simulate depth of     #
# field around the chosen point.                  #
#                                                 #
# INPUT:                                          #
#     - template: the middlemost image of the vid #
#     - xt: the leftmost coord of the template    #
#     - yt: the topmost coord of the template     #
#     - video: a sequence of images               #
# OUTPUT:                                         #
#     - img: the refocused image about the        #
#           template patch                        #
###################################################
def refocusUnstructured(template, xt, yt, video):
    img = np.zeros_like(video[0])
    count = 0

    for f in video:
        print(count)
        count += 1
        sx, sy = normalizedCrossCorrelation(f, template, xt, yt)

        sx = sx - 25
        sy = sy - 25
        imR = f[:, :, 0]
        imG = f[:, :, 1]
        imB = f[:, :, 2]
        
        iR = interp(imR, sy, sx)
        iG = interp(imG, sy, sx)
        iB = interp(imB, sy, sx)

        shifted = np.dstack((iR, iG, iB)).astype(np.uint16)

        img = (img + shifted)

        # if count == 10 or count == 20 or count == 30:
        #     # plt.imshow(f)
        #     plt.imsave('./res/original' + str(count) + '.png', f)
        #     # plt.show()
        #     # plt.imshow(shifted)
        #     plt.imsave('./res/shifted' + str(count) + '.png', shifted.astype(np.uint8))
        #     # plt.show()


    img = img / (count*255)

    return img


#################### MAIN: RUN ####################
# Put the functions above into practice to fully  #
# analyze and run through a video.                #
#                                                 #
# INPUT:                                          #
#     - None.                                     #
# OUTPUT:                                         #
#     - None.                                     #
###################################################
def run():
    # v = readVid('./data/appa.MOV')
    v = readVid('./data/appa2.MOV')
    plt.imshow(v[19])
    # plt.imsave('./res/middleImg.png', v[19])
    plt.show()

    template = v[19]

    plt.imshow(template[720:770, 1050:1100, :])
    plt.imsave('./res/templatepatch2.png', template[720:770, 1050:1100, :])
    plt.show()
    plt.imshow(template[530:580, 160:210, :])
    plt.imsave('./res/templatepatch2.png', template[530:580, 160:210, :])
    plt.show()
    plt.imshow(template[605:655, 310:360, :])
    plt.imsave('./res/templatepatch3.png', template[605:655, 310:360, :])
    plt.show()

    r1 = refocusUnstructured(template, 720, 1050, v)
    plt.imshow(r1)
    plt.imsave('./res/apparesult.png', r1)
    plt.show()

    r2 = refocusUnstructured(template, 530, 160, v)
    plt.imshow(r2)
    plt.imsave('./res/tabletresult.png', r2)
    plt.show()

    r3 = refocusUnstructured(template, 605, 310, v)
    plt.imshow(r3)
    plt.imsave('./res/mousetresult.png', r3)
    plt.show()
run()