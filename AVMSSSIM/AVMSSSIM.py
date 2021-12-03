import numpy as np
from MS_SSIM import MS_SSIM
from MS_SSIM_1D import MS_SSIM_1D

def AVMSSSIM(ref_video, dis_video, ref_audio, dis_audio):
    frame_num = ref_video.shape[2]
    mssim_frame = np.zeros((frame_num,))
    for i in range(frame_num):
        ref_frame = ref_video[:, :, i]
        dis_frame = dis_video[:, :, i]
        mssim_frame[i] = MS_SSIM(ref_frame, dis_frame)
    msssim_video = np.mean(mssim_frame)
    
    if ref_audio.shape[1] == 2:
        ref_audio = ref_audio[:, 0]
    if dis_audio.shape[1] == 2:
        dis_audio = dis_audio[:, 0]

    msssim_audio = MS_SSIM_1D(ref_audio, dis_audio)
    
    weight = 0.95
    AVSSSIM_score = msssim_video ** weight * msssim_audio ** (1-weight)
    return AVSSSIM_score