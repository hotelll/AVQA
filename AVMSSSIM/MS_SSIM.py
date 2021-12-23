from pytorch_msssim import ms_ssim
import torch

def MS_SSIM(reference, distorted):
    reference = torch.from_numpy(reference)
    distorted = torch.from_numpy(distorted)
    reference = reference.reshape(1, 1, 1080, 1920)
    distorted = distorted.reshape(1, 1, 1080, 1920)
    return ms_ssim(reference, distorted, data_range=255, size_average=False)
