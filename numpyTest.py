import cv2
import numpy as np


# 이차원 NumPy 배열 생성
two_dim_array = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

# 중복된 값 제거하고 일차원 배열로 합치기
one_dim_array = np.unique(two_dim_array).ravel()
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[1, 2, 3], [13, 14, 15]]])
unique_values_3d = np.unique(arr3d)
# print(unique_values_3d)
# print(one_dim_array)



# 3차원 배열 생성
arr3d = np.array([[[1, 2], [4, 6]], [[7, 9], [10, 12]], [[1, 3], [13,  15]]])

# 3차원 배열을 2차원 배열로 변환
arr2d = arr3d.reshape(-1, arr3d.shape[-1])

print("3차원 배열:")
print(arr3d)

print("2차원 배열:")
print(arr2d)