import numpy as np
from utils.utils import conv1d


def GMSD_1D(reference, distorted):
    down_step = 64
    ave_kernel = np.ones(down_step) / down_step
    reference = conv1d(reference, ave_kernel, 'valid')
    distorted = conv1d(distorted, ave_kernel, 'valid')
    
    # down-sampling
    distorted = distorted[::down_step]
    reference = reference[::down_step]
    
    dx = np.array([1, 0, -1])
    grad1 = conv1d(reference, dx, 'valid')
    grad2 = conv1d(distorted, dx, 'valid')
    gm1 = np.abs(grad1)
    gm2 = np.abs(grad2)
    quality_map = (2 * gm1 * gm2) / (gm1**2 + gm2**2)
    gmsd_1d_score = np.nanstd(quality_map, ddof=1)
    return gmsd_1d_score