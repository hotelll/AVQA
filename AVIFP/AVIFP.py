import numpy as np
from AVIFP.VIFP import VIFP
from AVIFP.VIFP_1D import VIFP_1D
from tqdm import tqdm

def AVIFP(ref_video, dis_video, ref_audio, dis_audio):
    frame_num = ref_video.shape[2]
    vifp_frame = np.zeros(frame_num, dtype=np.float64)
    for i in tqdm(range(frame_num)):
        ref_frame = ref_video[:, :, i]
        dis_frame = dis_video[:, :, i]
        vifp_frame[i] = VIFP(ref_frame, dis_frame)
    vifp_video = np.mean(vifp_frame)
    
    if ref_audio.shape[1] == 2:
        ref_audio = ref_audio[:, 0]
    
    if dis_audio.shape[1] == 2:
        dis_audio = dis_audio[:, 0]
    
    vifp_audio = VIFP_1D(ref_audio, dis_audio)
    
    weight = 0.7
    AVIFP_score = vifp_video ** weight * vifp_audio ** (1 - weight)
    return AVIFP_score