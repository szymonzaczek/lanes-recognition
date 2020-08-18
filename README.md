# Computer vision application: lanes recognition - differantiation between the current and the opposite directions

<p align="center">
    <img src="https://github.com/szymonzaczek/lanes-recognition/blob/master/lanes_recognition_lighter.gif?raw=true" alt="Lanes recognition: differantiation between the current and the opposite directions">
</p>

**This repository contains Python code for detecting road lanes and recognizing lanes that are the same as our current direction or the lanes that are in the opposite direction**

**Technologies/libraries used:**
- computer vision - opencv 
- matrices operations - numpy
- visualisations - matplotlib

**Walkthrough:**
- (OPTIONAL) play clip from car's camera using play_video.py
- save individual frames from the car's camera footage using save_frames.py
- running the actual lanes recognition program - lane_recognition.py:
  - sorting files with '.jpg' extension
  - lanes detection and recognition algorithm (operation is performed on each image)
    - enhancing blacks on the image
    - converting image to grayscale
    - enhancing the contrast of the grayscale image
    - blurring the image
    - detecting edges using sobel operator algorithm
    - applying region of interest to the image
    - detecting lines on the processed image
    - applying detected lines to the original image
    - saving the processed image
  - combining processed images to one '.gif' file - jpg_to_animated_gif.py


**Files description:**
- clip.mp4 - raw footage from car's camera
- jpg_to_animated_gif.py - Python code for converting resultant jpg files to one animated GIF file
- lane_recognition.py - Python code for the lanes recognition app
- lanes_recognition_lighter.gjf - the GIF file presenting the result of the app
- play_video.py - Python code for playing mp4 file
- save_frames.py - Python code for converting raw footage to individual jpg files (stored in images_clip directory)


