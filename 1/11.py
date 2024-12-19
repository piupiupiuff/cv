import cv2
import numpy as np

img = cv2.imread('f2.jpg')
height, width, channels = img.shape

# convert image into b-w
def black_white(img):
    bw_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = img[i][j]
            gray = (int(b) + int(g) + int(r)) / 3  # Преобразование в int предотвращает переполнение
            bw_img[i][j] = int(gray)
    return bw_img

# method Otsu
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

# create binary mask for image
def make_mask(t, blacknwhite_img):
    for i in range(height):
        for j in range(width):
            if blacknwhite_img[i, j] > t:
                blacknwhite_img[i, j] = 0
            else:
                blacknwhite_img[i, j] = 1

    return bw_img

bw_img = black_white(img)
t = otsu_method(bw_img)
img_mask = np.array(make_mask(t, bw_img) * 255, dtype=np.uint8)
m_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
mask_img = cv2.bitwise_and(img, m_color)
cv2.imwrite('finalf2.png', mask_img)
