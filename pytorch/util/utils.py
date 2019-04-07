import numpy as np
import scipy.stats as st
import cv2
import time
import os
import glob

def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2,sigma+interval/2,size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])
    return out_filter


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_rect_mask(im_size, mask_size, margin=8, rand_mask=True):
    mask = np.zeros((im_size[0], im_size[1])).astype(np.float32)
    if rand_mask:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = np.random.randint(margin, im_size[0] - sz0 - margin)
        of1 = np.random.randint(margin, im_size[1] - sz1 - margin)
    else:
        sz0, sz1 = mask_size[0], mask_size[1]
        of0 = (im_size[0] - sz0) // 2
        of1 = (im_size[1] - sz1) // 2
    mask[of0:of0+sz0, of1:of1+sz1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[of0, sz0, of1, sz1]], dtype=int)
    return mask, rect


def generate_stroke_mask(im_size, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask


def generate_mask(type, im_size, mask_size):
    if type == 'rect':
        return generate_rect_mask(im_size, mask_size)
    else:
        return generate_stroke_mask(im_size), None


def getLatest(folder_path):
    files = glob.glob(folder_path)
    file_times = list(map(lambda x: time.ctime(os.path.getctime(x)), files))
    return files[sorted(range(len(file_times)), key=lambda x: file_times[x])[-1]]
