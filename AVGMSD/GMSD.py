import numpy as np
from utils.utils import conv2d

def GMSD(reference, distorted):
    T=170
    down_step = 2
    # Prewitt operator
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]] / 3, 
                  dtype=np.float64)
    dy = dx.transpose()
    
    # average filter
    ave_kernel = np.array([[0.25, 0.25], 
                           [0.25, 0.25]])
    distorted = conv2d(distorted, ave_kernel, 'same')
    reference = conv2d(reference, ave_kernel, 'same')
    
    # down-sampling
    distorted = distorted[::down_step, ::down_step]
    reference = reference[::down_step, ::down_step]
    
    grad1 = conv2d(reference, dx, 'same')**2 + conv2d(reference, dy, 'same')**2
    grad2 = conv2d(distorted, dx, 'same')**2 + conv2d(distorted, dy, 'same')**2

    quality_map = (2 * np.sqrt(grad1) * np.sqrt(grad2) + T) / (grad1 + grad2 + T)
    gmsd_score = np.std(quality_map, ddof=1)
    return gmsd_score