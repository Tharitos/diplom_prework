import glob
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
first = cv.imread('img_2/DSC02694.JPG', 0)
second = cv.imread('img_2/DSC02692.JPG', 0)
fig, ax = plt.subplots(1,2, figsize=(15,15))
ax[0].imshow(first, cmap='gray')
ax[1].imshow(second, cmap='gray')
# Инициализация SIFT детектора
sift = cv.SIFT_create()
# Определение ключевых точек SIFT
kp1, des1 = sift.detectAndCompute(first, None)
kp2, des2 = sift.detectAndCompute(second, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# фильрация Лоу
matchesLowe = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        matchesLowe.append(m)
left_pts = np.float32([kp1[m.queryIdx].pt for m in matchesLowe])
right_pts = np.float32([kp2[m.trainIdx].pt for m in matchesLowe])
# Фундаментальная матрица
F, mask = cv.findFundamentalMat(left_pts, right_pts, cv.FM_RANSAC)
left_pts = left_pts[mask.ravel() == 1]
right_pts = right_pts[mask.ravel() == 1]
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
    return img1color
lines1 = cv.computeCorrespondEpilines(right_pts.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
lines2 = cv.computeCorrespondEpilines(left_pts.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
img1_lines = drawlines(first, second, lines1, left_pts, right_pts)
img2_lines = drawlines(second, first, lines2, right_pts, left_pts)
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(img1_lines, 'gray')
ax[1].imshow(img2_lines, 'gray')
h1, w1 = first.shape
h2, w2 = second.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(left_pts), np.float32(right_pts), F, imgSize=(w1, h1))
img1_rec = cv.warpPerspective(first, H1, (w1, h1))
img2_rec = cv.warpPerspective(second, H2, (w2, h2))

fig, ax = plt.subplots(1, 2, figsize=(15,15))
ax[0].imshow(img1_rec, 'gray')
stereo = cv.StereoBM_create(numDisparities=48, blockSize=5)
disparity = stereo.compute(img1_rec,img2_rec)
plt.figure(figsize = (15, 15))
plt.imshow(disparity,'gray')
plt.show()
ax[1].imshow(img2_rec, 'gray')