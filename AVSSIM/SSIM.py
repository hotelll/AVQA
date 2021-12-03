from pytorch_msssim import ssim

def SSIM(reference, distorted):
    return ssim(reference, distorted, data_range=255, size_average=False)
