# Assignment 1: Image processing pipeline

This project contains a custom image processing pipeline that takes a RAW image file, converts it to a TIFF using the below `dcraw` commands, then conducts linearization, demosaicing, white balancing, color space correction, and brightness/gamma correction. 

To run this code, first run the following `dcraw` commands on your image:

`dcraw -4 -d -v -w -T <RAW_filename>`

Take note of the darkness, saturation, and multiplier values, then delete the resulting TIFF and regenerate it with the following command:

`dcraw -4 -D -T <RAW_filename>`

The rest of the process is conducted in `imagepipeline.py`. To use your own image, update `main` with the path to your TIFF (line 227) and the darkness, saturation, and red-green-blue multiplier values (lines 230 - 234). Depending on your image, you may need to adjust the chosen Bayer pattern image in `main` as well.
