# Image-Matching
OpenCV-based feature matching with FLANN and SIFT to locate query images in larger scenes. Robust to rotation, scale, and occlusion. Includes template matching comparisons.

## Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `matplotlib`

```sh
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
```

### Step 2: Download Images

We need to Download the images from my `Github` repository or you can download your own sets.

```sh
!wget https://raw.githubusercontent.com/AsadiAhmad/Image-Matching/main/Pictures/ps5_games.jpg -O ps5_games.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Image-Matching/main/Pictures/gost_of_tsushima.jpg -O gost_of_tsushima.jpg
```

### Step 3: Load Images

We need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`.

```sh
query_image = cv.imread('gost_of_tsushima.jpg')
target_image = cv.imread('ps5_games.jpg')
```

<div display=flex align=center>
  <img src="/Pictures/Colors/HSV.jpg" width="400px"/>
  <img src="/Pictures/Colors/HSV.jpg" width="400px"/>
</div>

