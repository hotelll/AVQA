import numpy as np
from utils.utils import gaussian_kernel, conv1d

def MS_SSIM_1D(reference, distorted):
    win_size = 11
    win = gaussian_kernel([win_size, 1], 1.5).squeeze(1)
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    method = 'product'
    
    downsample_filter = np.array([0.5, 0.5])
    
    mssim_array = np.zeros((level, ))
    mcs_array = np.zeros((level, ))
    
    for l in range(level):
        mssim_array[l], mcs_array[l] = ssim_index_new(reference, distorted, win)
        filtered_reference = conv1d(reference, downsample_filter, 'same')
        filtered_distorted = conv1d(distorted, downsample_filter, 'same')
        del reference, distorted
        reference = filtered_reference[::2]
        distorted = filtered_distorted[::2]
    
    if method == 'product':
        MSSIM_1D_score = np.power(mcs_array[: level-1], weight[: level-1]).prod() \
                        * np.power(mssim_array[level-1], weight[level-1])
    else: # method = 'sum'
        MSSIM_1D_score = (mcs_array[: level-1] * weight[: level-1]).sum() \
                        + mssim_array[level-1] * weight[level-1]
    return MSSIM_1D_score

def ssim_index_new(reference, distorted, win):
    mu1 = conv1d(reference, win, 'valid')
    mu2 = conv1d(distorted, win, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = conv1d(reference * reference, win, 'valid') - mu1_sq
    sigma2_sq = conv1d(distorted * distorted, win, 'valid') - mu2_sq
    sigma12   = conv1d(reference * distorted, win, 'valid') - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2) * (2 * sigma12)) \
        / ((mu1_sq + mu2_sq) * (sigma1_sq + sigma2_sq))
    cs_map = (2 * sigma12) / (sigma1_sq + sigma2_sq)
    
    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)
    return mssim, mcs