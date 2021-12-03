import numpy as np
from VIFP_feat import VIFP_feat
from VIFP_1D_feat import VIFP_1D_feat

def AVMAF(ref_video, dis_video, ref_audio, dis_audio):
    frame_num = ref_video.shape[2]
    vifp_feat_frame = np.zeros((4, 4))
    for i in range(frame_num):
        ref_frame = ref_video[:, :, i]
        dis_frame = dis_video[:, :, i]
        vifp_feat_frame[i, :] = VIFP_feat(ref_frame, dis_frame)
    vifp_feat_video = np.mean(vifp_feat_frame, 0)
    
    # measure the adjacent frame difference
    diff_frame = np.zeros(frame_num-1)
    for i in range(frame_num-1):
        diff = dis_video[:, :, i] - dis_video[:, :, i+1]
        diff_frame[i, 0] = np.mean(np.abs(diff))
    diff_video = np.mean(diff_frame)
    
    # measure the audio quality
    if ref_audio.shape[1] == 2:
        ref_audio = ref_audio[:, 0]
    if dis_audio.shape[1] == 2:
        dis_audio = dis_audio[:, 1]
    
    vifp_feat_audio = VIFP_1D_feat(ref_audio, dis_audio)
    
    AVMAF_feat = np.array([vifp_feat_video, diff_video, vifp_feat_audio])
    return AVMAF_feat
    
    