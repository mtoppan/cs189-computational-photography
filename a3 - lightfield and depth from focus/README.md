# Macy Toppan
# Assignment 3
For greater detail on any function mentioned below, please see the annotations at the top of each function within the code itself.
## Q1
All of the content for the first task of the assignment is contained in 'lightfield.py'. This includes several functions:
* 'initials'
* 'subApertureViews'
* 'maxAperture'
* 'interp'
* 'reFocus'
* 'gammaDecoding'
* 'focalStack'
* 'allInFocus'

... all which are run by 'run'. To modify the running of the function at all, any changes may be made within 'run'. This includes modifying the depths (by changing 'ds'), changing the input image, etc.
To run the assignment, it may simply be called by running it as you normally would a function within VS code. 'run' is called within the file, thus it should proceed normally and print updates and images as it goes, saving important outputs to the 'res' folder.
## BONUS 1
This part of the assignment was not included in full, but is mentioned in that the alternate apertures (circle and square) were implemented as options. To use them, modify the following line in 'run' in 'lightfield.py': 

    stack = focalStack(L, ds, 'max', 1)
... s.t. 'max' is substituted for 'circle' or 'square', and '1' is substituted for some other number of your choosing, to determine the number of apertures tested. 

All functions relating to this are in the functions 'circularAperture' and 'squareAperture'.

## Q2
All of the content for the first task of the assignment is contained in 'unstructured.py'. This includes several functions:
* 'readVid'
* 'normalizedCrossCorrelation'
* 'interp'
* 'refocusUnstructured'

... all which are run by 'run'. To modify the running of the function at all, any changes may be made within 'run'. This is limited to changing the input video and the template patch— identifying the leftmost and topmost corner of the desired region (to be set at a size of 50x50)— which are determined by the user through reading the image.
gInput was attempted, but (perhaps due to computer issues) returned unreliable values, and so this alternate was used instead.
To run the assignment, it may simply be called by running it as you normally would a function within VS code. 'run' is called within the file, thus it should proceed normally and print updates and images as it goes, saving important outputs to the 'res' folder.
This file also calls on resources from previous assignments, specifically 'loadVid.py' (given in Computer Vision 23W) and 'cp_hw2.py' (given in Assignment 2 of Computational Photography and used here for its lRGB2XYZ function).