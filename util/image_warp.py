from PIL import Image
import cv2
import numpy as np


def crop2_43(img: Image.Image):
    img_reshaper = ImageReshaper(img)
    return img_reshaper.get_reshaped()


class ImageReshaper:
    def __init__(self, img: Image.Image):
        self.img = img
        self.trans, self.inv_trans = crop2_43_trans(self.img)
        w, h = self.img.size
        self.trans_mask = self.get_trans_mask(self.inv_trans, [h, w])

    def get_reshaped(self):
        img = np.array(self.img)
        new_h = 1024
        new_w = 768
        trans_img = cv2.warpAffine(img, self.trans, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
        return Image.fromarray(trans_img)

    def back2rawSahpe(self, img):
        raw_img = np.array(self.img)
        new_img = np.array(img)
        w, h = self.img.size
        raw_new_img = self.roi2raw(new_img, self.inv_trans, [h, w])
        composed = raw_img.copy()
        composed[self.trans_mask] = raw_new_img[self.trans_mask]
        return composed

    def roi2raw(self, img, trans, raw_shape):
        trans_img = cv2.warpAffine(img, trans, (raw_shape[1], raw_shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE,  # cv2.BORDER_CONSTANT,
                                   # borderValue=(0, 0, 0)
                                   )
        return trans_img

    def get_trans_mask(self, inv_trans, raw_shape):
        mask = np.ones([1024, 768]).astype(np.uint8)
        roi_mask = cv2.warpAffine(mask, inv_trans, (raw_shape[1], raw_shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0
                                  )
        roi_mask = roi_mask.astype(bool)
        return roi_mask


def crop2_43_trans(img: Image.Image):
    raw_h, raw_w = img.size
    src = np.zeros([3, 2], np.float32)
    if 3 * raw_h > 4 * raw_w:  # too tall
        delta = (raw_h - raw_w * (4 / 3)) / 2
        src[0, :] = np.array([0 + delta, 0], np.float32)
        src[1, :] = np.array([raw_h - delta, 0], np.float32)
        src[2, :] = np.array([raw_h - delta, raw_w], np.float32)
    else:  # too wide
        delta = (raw_w - raw_h * (3 / 4)) / 2
        src[0, :] = np.array([0, 0 + delta], np.float32)
        src[1, :] = np.array([raw_h, 0 + delta], np.float32)
        src[2, :] = np.array([raw_h, raw_w - delta], np.float32)

    dst = np.zeros([3, 2], np.float32)
    dst[0, :] = np.array([0, 0], np.float32)
    dst[1, :] = np.array([1024, 0], np.float32)
    dst[2, :] = np.array([1024, 768], np.float32)
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return trans, inv_trans
