import numpy as np

from skimage.transform import resize
import cv2
from scipy.ndimage import gaussian_filter
import random
import math

def affine_transformation(img, m=1.0, s=.2, border_value=None):
    h, w = img.shape[0], img.shape[1]
    src_point = np.float32([[w / 2.0, h / 3.0],
                            [2 * w / 3.0, 2 * h / 3.0],
                            [w / 3.0, 2 * h / 3.0]])
    random_shift = m + np.random.uniform(-1.0, 1.0, size=(3,2)) * s
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if border_value is None:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(w, h), borderValue=float(border_value))
    return warped_img

def image_resize(img, height=None, width=None):

    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img


def centered(word_img, tsize, centering=(.5, .5), border_value=None):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    try:
        word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)
    except:
        word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw, (0,0)), 'constant', constant_values=border_value)
        word_img = np.pad(word_img[ys:ye, xs:xe, 0], (padh, padw), 'constant', constant_values=border_value)
    return word_img

# Additions

# adjusts the brightness of the image up or down
def relight(img,v=0.5):
    adj = np.exp(np.random.normal())
    img = np.power(img,adj)
    return img

# returns pink noise -- from https://stackoverflow.com/questions/70085015/how-to-generate-2d-colored-noise
# def pink_noise(shape):
#     whitenoise = np.random.uniform(0, 1, shape)
#     ft_arr = np.fft.fftshift(np.fft.fft2(whitenoise))
#     _x, _y = np.mgrid[0:ft_arr.shape[0], 0:ft_arr.shape[1]]
#     f = np.hypot(_x - ft_arr.shape[0] / 2, _y - ft_arr.shape[1] / 2)
#     pink_ft_arr = ft_arr / f
#     pink_ft_arr = np.nan_to_num(pink_ft_arr, nan=0, posinf=0, neginf=0)
#     pinknoise = np.fft.ifft2(np.fft.ifftshift(pink_ft_arr)).real
#     return pinknoise

# adds normal noise, possibly with gaussian smoothing
def addnoise(img, scale=0.2, sm=None):
    noise = np.random.normal(scale=scale, size=np.shape(img))
    if sm != None:
        noise = gaussian_filter(noise, sm, mode='wrap');
    return np.clip(img+np.float32(noise),0,1)
        
# trims some of the extra off the left and right sides of the image (assumes raw images are already extended)
def trimSides(img, amts=[1]):
    ltrim = random.choice(amts)
    rtrim = random.choice(amts)
    nrow = img.shape[-2]
    ncol = img.shape[-1]
    if len(img.shape)==2:
        return image_resize(img[:,round(ltrim*nrow):ncol-round(rtrim*nrow)], height=int(1.0 * nrow), width=int(1.0 * ncol))
    else:
        return np.stack([image_resize(ply[:,round(ltrim*nrow):ncol-round(rtrim*nrow)], height=int(1.0 * nrow), width=int(1.0 * ncol)) for ply in img],0)
