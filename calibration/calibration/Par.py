import glob
import numpy as np
import cv2 as cv
# Внутренняя
K = np.array([[6.05304389e+03, 0.00000000e+00, 2.54106629e+03],
 [0.00000000e+00, 6.03790773e+03, 1.30293219e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
from matplotlib import pyplot as plt
first = cv.imread('img_2/DSC02694.JPG', 0)
second = cv.imread('img_2/DSC02692.JPG', 0)
fig, ax = plt.subplots(1,2, figsize=(15,15))
ax[0].imshow(first, cmap='gray')
ax[1].imshow(second, cmap='gray')
# Ключевые точки
# Инициализация SIFT детектора
sift = cv.SIFT_create()
# Определение ключевых точек SIFT
kp1, des1 = sift.detectAndCompute(first, None)
kp2, des2 = sift.detectAndCompute(second, None)
# Сопоставление на основе FLANN параметров
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
left_pts = np.float32([kp1[m.queryIdx].pt for m in matchesLowe]).reshape(-1, 1, 2)
right_pts = np.float32([kp2[m.trainIdx].pt for m in matchesLowe]).reshape(-1, 1, 2)
# Матрицы
E, _ = cv.findEssentialMat(left_pts, right_pts, K)
F, _ = cv.findFundamentalMat(left_pts, right_pts, cv.FM_LMEDS)
# Точки
left_cupboard  = np.array([[533, 1523], [1643, 1562], [1608, 4312], [570, 4483]])
right_cupboard = np.array([[366, 1526], [1363, 1486], [1379, 4182], [447, 4070]])
left_door = np.array([[113, 1117], [167, 4310]])
right_door = np.array([[352, 1168], [434, 3892]])
white = (255, 255, 255)
black = (0, 0, 0)
for index in range(len(left_cupboard)):
    cv.circle(first,  tuple(left_cupboard[index]),  25, white, -1)
    cv.circle(second, tuple(right_cupboard[index]), 25, white, -1)

for index in range(len(left_door)):
    cv.circle(first,  tuple(left_door[index]),  25, black, -1)
    cv.circle(second, tuple(right_door[index]), 25, black, -1)
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(first, cmap='gray')
ax[1].imshow(second, cmap='gray')
# Переход к мировой системе координат
# Первая матрица проекций
rt1 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])
pr1 = np.dot(K, rt1)
# Вторая матрица проекций
_, R, T, _ = cv.recoverPose(E, left_pts, right_pts, K)
rt2 = np.concatenate((R, T), axis=1)
pr2 = np.dot(K, rt2)

def triangulate(left, right, i):
    res = cv.triangulatePoints(pr1, pr2, left[i], right[i])
    return res[:3] / res[-1]


global_cupboard = []
global_door = []

for i in range(len(left_cupboard)):
    global_cupboard.append(triangulate(left_cupboard.astype(float), right_cupboard.astype(float), i))

for i in range(len(left_door)):
    global_door.append(triangulate(left_door.astype(float), right_door.astype(float), i))
    
global_cupboard = np.array(global_cupboard)
global_door = np.array(global_door)
# Пусть высота двери 2 метра
scale = 2 / cv.norm(global_door[0] - global_door[1], cv.NORM_L2)
distances = np.array([scale * cv.norm(global_cupboard[i] - global_cupboard[(i + 1) % len(global_cupboard)], cv.NORM_L2) for i in range(len(global_cupboard))])
distances
scale