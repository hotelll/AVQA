from AVGMSM.GMSM import GMSM
from AVGMSM.GMSM_1D import GMSM_1D
from tqdm import tqdm
import numpy as np

def AVGMSM(ref_video, dis_video, ref_audio, dis_audio):
    frame_num = ref_video.shape[2]
    gmsm_frame = np.zeros(frame_num, dtype=np.float64)
    for i in tqdm(range(frame_num)):
        ref_frame = ref_video[:, :, i]
        dis_frame = dis_video[:, :, i]
        gmsm_frame[i] = GMSM(ref_frame, dis_frame)
    gmsd_video = np.mean(gmsm_frame)
    
    if ref_audio.shape[1] == 2:
        ref_audio = ref_audio[:, 0]
    
    if dis_audio.shape[1] == 2:
        dis_audio = dis_audio[:, 0]
    
    gmsd_audio = GMSM_1D(ref_audio, dis_audio)
    
    weight = 0.8
    AVGMSM_score = gmsd_video**weight * gmsd_audio**(1 - weight)
    return AVGMSM_score