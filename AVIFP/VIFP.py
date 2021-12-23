import numpy as np
from utils.utils import filter2d, gaussian_kernel

def VIFP(reference, distorted):
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0
    
    for scale in range(1, 5):
        N = 2**(5 - scale) + 1
        win = gaussian_kernel([N, N], N / 5.0)
        
        if scale > 1:
            reference = filter2d(win, reference, 'valid')
            distorted = filter2d(win, distorted, 'valid')
            # down-sampling
            distorted = distorted[::2, ::2]
            reference = reference[::2, ::2]
        
        mu1 = filter2d(win, reference, 'valid')
        mu2 = filter2d(win, distorted, 'valid')
        sigma1_sq = filter2d(win, reference * reference, 'valid') - mu1 * mu1
        sigma2_sq = filter2d(win, distorted * distorted, 'valid') - mu2 * mu2
        sigma12 = filter2d(win, reference * distorted, 'valid') - mu1 * mu2
        
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0
        
        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0
        
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps
        
        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    
    VIFP_score = num / den
    return VIFP_score