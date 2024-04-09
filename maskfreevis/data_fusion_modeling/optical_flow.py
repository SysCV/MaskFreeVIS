import cv2 as cv
import numpy as np


def extract_optical_flow_dense_matrix(prev_img: np.ndarray,
                                      next_img: np.ndarray) -> np.ndarray:
    """
    Extract dense optical flow feature between 2 images using Lucas-Kanade method.

    Args:
        prev_img (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
        next_img (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8

        return 
            (H,W,3) RGB dense optical flow feature in 0-255 range, can be either float or uint8
    """

    prev_gray_img = cv.cvtColor(prev_img, cv.COLOR_RGB2GRAY)
    prev_img_hsv_array = np.zeros_like(prev_img)
    prev_img_hsv_array[..., 1] = 255

    # TODO: calcOpticalFlowFarneback params will check.
    next_gray_img = cv.cvtColor(next_img, cv.COLOR_BGR2GRAY)
    flow_between_imgs = cv.calcOpticalFlowFarneback(prev_gray_img, next_gray_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv.cartToPolar(flow_between_imgs[..., 0], flow_between_imgs[..., 1])
    prev_img_hsv_array[..., 0] = angle * 180 / np.pi / 2
    prev_img_hsv_array[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    optical_flow_rgb_img = cv.cvtColor(prev_img_hsv_array, cv.COLOR_HSV2RGB)
    return optical_flow_rgb_img