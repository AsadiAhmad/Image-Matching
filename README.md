# Image-Matching
OpenCV-based feature matching with FLANN and SIFT to locate query images in larger scenes. Robust to rotation, scale, and occlusion. Includes template matching comparisons.

## FLANN Overview

FLANN (Fast Library for Approximate Nearest Neighbors) is an optimized algorithm for fast nearest-neighbor searches in high-dimensional spaces (like SIFT's 128D descriptors). It's much faster than brute-force matching for large datasets.

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
descriptor_query (descriptors): A NumPy array of shape (N, 128) where 
- N = Number of keypoints detected.
- 128 = The SIFT descriptor dimension (a 128-dimensional feature vector).

### Step 7: Calculate FLANN parameters

```sh
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptor_query, descriptor_target, k=2)
```

### Step 8: Refine matches

```sh
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ key_point_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ key_point_target[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = gray_query_image.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    result = cv.polylines(result,[np.int32(dst)],True,(0,255,0),3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
```

### Step 9: Show Result

```sh
plt.figure(figsize=[15,4])
plt.subplot(131),plt.imshow(query_image[...,::-1]),plt.title('Image1');
plt.subplot(132),plt.imshow(target_image[...,::-1]),plt.title('Image2');
plt.subplot(133),plt.imshow(result[...,::-1]),plt.title('result');
```

<div display=flex align=center>
  <img src="/Pictures/result.jpg"/>
</div>

