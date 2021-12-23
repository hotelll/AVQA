from AVGMSD.AVGMSD import AVGMSD
from AVGMSM.AVGMSM import AVGMSM
from AVIFP.AVIFP import AVIFP
from AVMSSSIM.AVMSSSIM import AVMSSSIM
from utils.utils import read_video, read_audio
from scipy.stats import spearmanr, pearsonr
import json


labels = {}
with open("test.json", "r") as f:
    labels = json.load(f)

REFERENCE_PATH = "data/LIVE-SJTU_AVQA/Reference/"
DISTORTED_PATH = "data/LIVE-SJTU_AVQA/Distorted/"

predicts = []
ground_truth = []
index = 1
test_num = len(labels)

for info in labels.values():
    ref_video_path = REFERENCE_PATH + info["ref_video"]
    ref_audio_path = REFERENCE_PATH + info["ref_audio"]
    dis_video_path = DISTORTED_PATH + info["dis_video"]
    dis_audio_path = DISTORTED_PATH + info["dis_audio"]
    
    frame_num  = int(info["frame_num"])
    ref_video = read_video(ref_video_path, frame_num)
    ref_audio = read_audio(ref_audio_path) / 32768
    dis_video = read_video(dis_video_path, frame_num)
    dis_audio = read_audio(dis_audio_path) / 32768
    
    print("---------------------- video {} / {} ----------------------".format(index, test_num))
    score = AVMSSSIM(ref_video, dis_video, ref_audio, dis_audio)
    predicts.append(score)
    ground_truth.append(info["MOS"])
    index += 1
    

res = {}
res["predict"] = predicts
res["gt"] = ground_truth
res = json.dumps(res)
f = open("res.json", "w")
f.write(res)
f.close()

print("SRCC:", spearmanr(predicts, ground_truth)[0])
print("PLCC:", pearsonr(predicts, ground_truth)[0])
