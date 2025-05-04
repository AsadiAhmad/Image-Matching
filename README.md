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
  <img src="/Pictures/gost_of_tsushima.jpg" width="400px"/>
  <img src="/Pictures/ps5_games.jpg" width="400px"/>
</div>

### Step 4: Create the copy of the target Image

```sh
result = target_image.copy()
```

### Step 5: Change Images Color Space into GrayScale

```sh
gray_query_image = cv.cvtColor(query_image, cv.COLOR_BGR2GRAY)
gray_target_image = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
```

### Step 6: Use SIFT for detecting keypoints

In this step we define the SIFT from OpenCV and find keypoints (we can call them keypoint fetures too).

Keypoint Detection:

- SIFT uses differences of Gaussians (DoG) to find scale-invariant points.
- Filters out low-contrast/keypoints on edges (using Hessian matrix).

Descriptor Generation:

- For each keypoint, it analyzes gradients in a 16x16 region around it.
- Divides the region into 4x4 sub-blocks → computes an 8-bin gradient histogram per block.
- Concatenates histograms into a 128-element vector (the descriptor).

```sh
sift = cv.SIFT_create()

key_point_query, descriptor_query  = sift.detectAndCompute(gray_query_image, None)
key_point_target, descriptor_target = sift.detectAndCompute(gray_target_image, None)
```

key_point_query (keypoints): A list of KeyPoint objects, each representing a distinctive location in the image. each KeyPoint contains:
- pt: (x, y) coordinates of the keypoint.
- size: Scale of the keypoint.
- angle: Orientation (in degrees).
- response: Strength of the keypoint.
- 
descriptor_query (descriptors): A NumPy array of shape (N, 128) where 
- N = Number of keypoints detected.
- 128 = The SIFT descriptor dimension (a 128-dimensional feature vector).

