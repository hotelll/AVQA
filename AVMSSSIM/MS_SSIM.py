from pytorch_msssim import ms_ssim
import torch

def MS_SSIM(reference, distorted):
    reference = torch.from_numpy(reference)
    distorted = torch.from_numpy(distorted)
    
    return ms_ssim(reference, distorted, data_range=255, size_average=False)
