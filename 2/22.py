import cv2
import numpy as np

img = cv2.imread('2.jpg')
height, width, channels = img.shape

def black_white(img):
    bw_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = img[i][j]
            gray = (int(b) + int(g) + int(r)) / 3
            bw_img[i][j] = gray
    return bw_img

#Otsu Method
def otsu_method(bw_img):
    H = np.zeros(256)
    for i in range(height):
        for j in range(width):
            one_pixel = bw_img[i, j]
            H[one_pixel] = H[one_pixel] + 1

    probabilities_list = H/(height * width)
    variances_list = []
    for i in range(0, 256):
        q_1 = 0
        for x in range(i):
            q_1 = q_1 + probabilities_list[x]

        q_2 = 0
        for x in range(i + 1, 256):
            q_2 = q_2 + probabilities_list[x]

        if q_1 == 0 or q_2 == 0: continue

        avg1 = 0
        for x in range(i):
            avg1 = avg1 + (x * probabilities_list[x]) / q_1

        avg2 = 0
        for x in range(i + 1, 256):
            avg2 = avg2 + (x * probabilities_list[x]) / q_2

        variance_1 = 0
        for x in range(i):
            variance_1 = variance_1 + ((x - avg1) ** 2) * (probabilities_list[x] / q_1)
        variance_1 = variance_1 ** 2

        variance_2 = 0
        for x in range(i + 1, 256):
            variance_2 = variance_2 + ((x - avg2) ** 2) * (probabilities_list[x] / q_2)
        variance_2 = variance_2 ** 2

        variances_list.append((q_1 * variance_1 + q_2 * variance_2) ** 2)
        min_variance = min(variances_list)
        t_val = variances_list.index(min_variance)

    return t_val

def apply_otsu_threshold(t, bw_img):
    otsu_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            otsu_img[i, j] = 255 if bw_img[i, j] > t else 0
    return otsu_img

#Niblack Method
def niblack_method(img, k, w_s=15):

    h, w = img.shape
    dop_img = np.pad(img, ((w_s // 2, w_s // 2), (w_s // 2, w_s // 2)), 'reflect')
    niblack_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            x, y = i + w_s // 2, j + w_s // 2
            window = dop_img[x - w_s // 2:x + w_s // 2 + 1, y - w_s // 2:y + w_s // 2 + 1]
            l_br = np.mean(window)
            l_std = np.std(window)

            threshold = l_br + k * l_std
            niblack_img[i, j] = 0 if img[i, j] < threshold else 255

    return niblack_img

#Sauvola Method
def sauvola_method(img, w_s=15, R=128, k=0.2):
    h, w = img.shape
    dop_img = np.pad(img, ((w_s // 2, w_s // 2), (w_s // 2, w_s // 2)), 'reflect')
    sauvola_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            x, y = i + w_s // 2, j + w_s // 2
            window = dop_img[x - w_s // 2:x + w_s // 2 + 1, y - w_s // 2:y + w_s // 2 + 1]
            l_br = np.mean(window)
            l_std = np.std(window)

            threshold = l_br * (1 + k * ((l_std / R) - 1))
            sauvola_img[i, j] = 0 if img[i, j] < threshold else 255

    return sauvola_img

#Сhristian Method
def christian_method(img, w_s=15, k=0.2):
    h, w = img.shape
    min_br = np.min(img)
    max_std = 0

    for i in range(w_s // 2, h - w_s // 2):
        for j in range(w_s // 2, w - w_s // 2):
            window = img[i - w_s // 2:i + w_s // 2 + 1, j - w_s // 2:j + w_s // 2 + 1]
            window_std = np.std(window)
            if window_std > max_std:
                max_std = window_std

    dop_img = np.pad(img, ((w_s // 2, w_s // 2), (w_s // 2, w_s // 2)), 'reflect')
    christian_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            x, y = i + w_s // 2, j + w_s // 2
            window = dop_img[x - w_s // 2:x + w_s // 2 + 1, y - w_s // 2:y + w_s // 2 + 1]
            l_br = np.mean(window)
            l_std = np.std(window)

            threshold = (1 - k) * l_br + k * min_br + k * (l_std / max_std) * (l_br - min_br)
            christian_img[i, j] = 0 if img[i, j] < threshold else 255

    return christian_img


bw_img = black_white(img)

#Otsu Method
t = otsu_method(bw_img)
final_otsu_img = apply_otsu_threshold(t, bw_img)
cv2.imwrite('final_Otsu1.png', final_otsu_img)

#Niblack Method
niblack_img = niblack_method(bw_img, k=0.2, w_s=15)
cv2.imwrite('final_Niblack1.png', niblack_img)

#Sauvola Method
sauvola_img = sauvola_method(bw_img, k=0.2, w_s=15)
cv2.imwrite('final_Sauvola1.png', sauvola_img)

#Сhristian Method
christian_img = christian_method(bw_img, k=0.2, w_s=15)
cv2.imwrite('final_Christian1.png', christian_img)