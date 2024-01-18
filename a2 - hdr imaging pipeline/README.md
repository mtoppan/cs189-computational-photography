All code must be run from the `src` folder.
### To convert RAW images to .TIFF:
Compile `develop_raw.sh` with `chmod u+x develop_raw.sh`, then run with `./develop_raw.sh`. This should convert the RAW images for both door_stack and dali_stack (my personal photos) to .tiff files in their respective folders. Any output that may be necessary is piped into text files.
### To test hdr.py:
99% of the modifiers for `hdr.py` are in `demo.py`, where the user can change the image type, weight option, and merging method in the noted locations. The only things that should be updated in `hdr.py` are the exposure_start in `readImagesAndExposures` (which need only be changed if the user is bringing in images named differently from those used by me in this assignment) and the e value in `mergeExposureStack`. Note that, to counteract the possibility of errors from dividing by zero, I add that same tiny value to the denominator in all equations. This small changes prevents errors without harming results.
All of Q4 is included at the bottom of `demo.py`.
Past that, just run `demo.py` to see what `hdr.py` and the color correcting/whitebalancing/tonemapping functions can do! All images are saved to the `results` folder.
### Data
For the sake of not forcing you to download every massive file, I've included the RAW and rendered images of `door_stack` and `dali_stack` as separate zip files. If you intend to try them, please download the files, unzip the folders, and place them in the `data` folder, then run `develop_raw.sh`.