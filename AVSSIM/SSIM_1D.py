import numpy as np
from utils import conv1d, gaussian_kernel

def SSIM_1D(reference, distorted):
    win = gaussian_kernel((1, 60), 10).T
    mu1 = conv1d(reference, win, 'same')
    mu2 = conv1d(distorted, win, 'same')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu1
    sigma1_sq = conv1d(reference * reference, win, 'same') - mu1_sq
    sigma2_sq = conv1d(distorted * distorted, win, 'same') - mu2_sq
    sigma12 = conv1d(reference * distorted, win, 'same') - mu1_mu2
    ssim_map = ((2 * mu1_mu2) * (2 * sigma12)) \
                / ((mu1_sq + mu2_sq) * (sigma1_sq + sigma2_sq))
    mssim_1D = np.mean(ssim_map[:])
    return mssim_1D