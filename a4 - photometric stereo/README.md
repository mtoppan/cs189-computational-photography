# Assignment 4
### Macy Toppan
For greater detail on any function mentioned below, please see the annotations at the top of each function within the code itself.
### Q1: Photometric Stereo
All of the content for this assignment is contained in 'photometricStereo.py'. This includes several functions:

* 'initials'
* 'getNormals'
* 'showAlbedoAndNorms'
* 'showAlbedoAndNormsQ'
* 'enforceIntegrability'
* 'normalIntegration'
* 'showDepthAnd3D'
* 'calibratePhotometricStereo'

... all which are run by 'run'. To modify the running of the function at all, any changes may be made within 'run'. This includes providing a qualifier for initials to instruct it to use '.tif' or '.tiff', as the self-provided data uses the latter while the given inputs use the former. For this question, this argument need not be offered, or can be set as 'q=1'.
Additionally, the results for this portion of the assignment make use of Poisson integration presently  (calling on the given 'cp_hw4.py') but can be adjusted to use sFrankot instead by adjusting 'showDepthAnd3D' to use 'sFrankot' rather than 'sPoisson'. Finally, the user can change the 'g' values by manually altering 'g1' and/or 'g2' to provide a different matrix.
To run the assignment, it may simply be called by running it as you normally would a function within VS code. 'run' is called within the file, thus it should proceed normally, printing updates and showing images as it goes. As the different runs call on the same functions, to save the images the user must do so from the visualizer window themselves (to avoid overwriting).
### BONUS: DIY Stereo
This part of the assignment is included in 'photometricStereo.py' as well, and is also run by 'run'. No changes need to be made to visualize this. The results for this portion of the assignment make use of Frankot integration presently  (calling on the given 'cp_hw4.py') but can be adjusted to use sFrankot instead by adjusting 'showDepthAnd3D' to use 'sPoisson' rather than 'sFrankot'.
If new images are desired, the user can provide the path string in 'run'. Images must be entered as .tiffs, and can be converted from raw using the following bash command:

``find ./folder -name "*.cr2" -exec sh -c 'dcraw -v -T -4 -w -o 1 "$1" 2>&1' sh {} \; ``

This file also calls on resources from previous assignments, specifically 'cp_hw2.py' (given in Assignment 2 of Computational Photography and used here for its lRGB2XYZ function).