import numpy as np
from SSIM import SSIM
from SSIM_1D import SSIM_1D

def AVSSIM(ref_video, dis_video, ref_audio, dis_audio):
    frame_num = ref_video.shape[2]
    ssim_frame = np.zeros((frame_num,))
    for i in range(frame_num):
        ref_frame = ref_video[:, :, i]
        dis_frame = dis_video[:, :, i]
        ssim_frame[i] = SSIM(ref_frame, dis_frame)
    ssim_video = np.mean(ssim_frame)
    
    if ref_audio.shape[1] == 2:
        ref_audio = ref_audio[:, 0]
    if dis_audio.shape[1] == 2:
        dis_audio = dis_audio[:, 0]
    
    ssim_audio = SSIM_1D(ref_audio, dis_audio)
    
    weight = 0.95
    AVSSIM_score = ssim_video**weight * ssim_audio**(1-weight)
    return AVSSIM_score