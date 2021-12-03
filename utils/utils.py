import numpy as np
from scipy.signal import convolve2d
from scipy.io import wavfile
import sys

def conv1d(input, kernel, mode='same'):
    return np.convolve(input, kernel, mode=mode)

def conv2d(input, kernel, mode='same'):
    return convolve2d(input, kernel, mode=mode)

def filter2d(filter, input, mode='same'):
    return convolve2d(input, np.rot90(filter), mode=mode)

def gaussian_kernel(shape, sigma):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def read_video(file_path, shape=(1080, 1920)):
    '''
    usage: frame_list = yuv2frame(file_path)
    '''
    frame_list = np.zeros((1080, 1920, 336))
    height, width = shape
    frame_len = width * height * 3 // 2
    f = open(file_path, 'rb')
    shape = (int(height * 1.5), width)
    for i in range(336):
        raw = f.read(frame_len)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape((int(height*1.5), width))
        # 1080 * 1920
        frame_list[:, :, i] = yuv[: height, :]

    return frame_list
        
def read_audio(file_path):
    _, audio_list = wavfile.read(file_path)
    return audio_list
