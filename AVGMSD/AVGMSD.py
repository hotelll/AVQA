from AVGMSD.GMSD import GMSD
from AVGMSD.GMSD_1D import GMSD_1D
from tqdm import tqdm
import numpy as np

def AVGMSD(ref_video, dis_video, ref_audio, dis_audio):
    frame_num = ref_video.shape[2]
    gmsd_frame = np.zeros(frame_num, dtype=np.float64)
    for i in tqdm(range(frame_num)):
        ref_frame = ref_video[:, :, i]
        dis_frame = dis_video[:, :, i]
        gmsd_frame[i] = GMSD(ref_frame, dis_frame)
    gmsd_video = np.mean(gmsd_frame)
    if ref_audio.shape[1] == 2:
        ref_audio = ref_audio[:, 0]
    
    if dis_audio.shape[1] == 2:
        dis_audio = dis_audio[:, 0]
    
    gmsd_audio = GMSD_1D(ref_audio, dis_audio)
    gmsd_video_nor = 1 - gmsd_video / 0.25
    gmsd_audio_nor = 1 - gmsd_audio / 0.4
    
    weight = 0.65
    AVGMSD_score = gmsd_video_nor ** weight * gmsd_audio_nor ** (1 - weight)
    return AVGMSD_score