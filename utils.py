import numpy as np
import cv2


def read_image_with_alpha(file_):
    img = cv2.imread(file_, cv2.IMREAD_UNCHANGED)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def blend(img1, img2):
    b1, g1, r1, alpha1 = cv2.split(img1)
    b2, g2, r2, alpha2 = cv2.split(img2)
    alpha = alpha2.astype(float) / 255
    b = b2 * alpha + b1 * (1 - alpha)
    g = g2 * alpha + g1 * (1 - alpha)
    r = r2 * alpha + r1 * (1 - alpha)
    alpha = (alpha1 + alpha2) // 2
    return cv2.merge(
        (b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8), alpha))
