import numpy as np
from PIL import Image
from math import ceil
import matplotlib.pyplot as plt

IMAGE = Image.open('../data/example.jpg')
FULL_IMAGE_MATRIX = np.array(IMAGE)

COLOR = (123, 234, 15)


def rgb_to_hsv_1(matrix):
    matrix = matrix.astype('float')

    r, g, b = matrix[:, :, 0], matrix[:, :, 1], matrix[:, :, 2]

    MAX = matrix.max(axis=2)
    Mis = matrix.argmax(axis=2)
    MIN = matrix.min(axis=2)

    C = MAX - MIN

    H = np.zeros(r.shape)
    H = np.where((Mis == 0), ((g - b) / C) % 6, H)
    H = np.where((Mis == 1), ((b - r) / C) + 2, H)
    H = np.where((Mis == 2), ((r - g) / C) + 4, H)

    H *= 60

    V = MAX

    Sv = np.where(V == 0, 0, C / V)
    HSV = np.dstack([H, Sv, V / 255])

    return HSV


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    C = v * s
    H = h / 60
    X = C * (1 - abs(H % 2 - 1))

    R = np.zeros(h.shape)
    R = np.where((H >= 0) & (H < 1) | (H >= 5) & (H < 6), C, R)
    R = np.where((H >= 1) & (H < 2) | (H >= 4) & (H < 5), X, R)

    G = np.zeros(h.shape)
    G = np.where((H >= 0) & (H < 1) | (H >= 3) & (H < 4), X, G)
    G = np.where((H >= 1) & (H < 3), C, G)

    B = np.zeros(h.shape)
    B = np.where((H >= 3) & (H < 5), C, B)
    B = np.where((H >= 2) & (H < 3) | (H >= 5) & (H < 6), X, B)

    m = v - C

    r, g, b = (R + m) * 255, (G + m) * 255, (B + m) * 255
    RGB = np.dstack((np.ceil(r), np.ceil(g), np.ceil(b)))
    RGB = np.asarray(RGB, 'int32')
    print(RGB)

    return RGB


#print(rgb_to_hsv_1(FULL_IMAGE_MATRIX))
print(FULL_IMAGE_MATRIX)
plt.imshow(FULL_IMAGE_MATRIX)
plt.show()
hsv_image = rgb_to_hsv_1(FULL_IMAGE_MATRIX)
plt.imshow(hsv_image)
plt.show()
converted_from_hsv = hsv_to_rgb(hsv_image)
# print(f"Converted: {converted_from_hsv}")
# print(f"Original{FULL_IMAGE_MATRIX}")
plt.imshow(converted_from_hsv)
plt.show()


#print(hsv_to_rgb(rgb_to_hsv_1(FULL_IMAGE_MATRIX)))
