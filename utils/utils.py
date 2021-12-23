import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from scipy.io import wavfile
import cv2



def conv1d(input, kernel, mode='same'):
    return np.convolve(input, kernel, mode=mode)

# def conv2d(input, kernel, mode='same'):
#     return convolve2d(input, kernel, mode=mode)
# def conv2d(input, kernel, mode='same'):
#     return convolve(input, kernel, mode='constant', cval=0.0)


def conv2d(input, kernel, mode='same'):
    return cv2.filter2D(input, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def filter2d(kernel, input, mode='same'):
    return cv2.filter2D(input, -1, np.rot90(kernel), borderType=cv2.BORDER_CONSTANT)


def gaussian_kernel(shape, sigma):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def read_video(file_path, frame_num):
    '''
    usage: frame_list = yuv2frame(file_path)
    '''
    frame_list = np.zeros((1080, 1920, frame_num))
    height, width = 1080, 1920
    frame_len = width * height * 3 // 2
    f = open(file_path, 'rb')
    shape = (int(height*1.5), width)
    for i in range(frame_num):
        raw = f.read(frame_len)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape(shape)
        # 1080 * 1920
        frame_list[:, :, i] = yuv[: height, :]

    return frame_list

  
def read_audio(file_path):
    _, audio_list = wavfile.read(file_path)
    return audio_list
