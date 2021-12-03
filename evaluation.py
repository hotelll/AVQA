from AVGMSD.AVGMSD import AVGMSD
from utils.utils import read_video, read_audio
from scipy.stats import spearmanr
import json
from tqdm import tqdm

labels = {}
with open("label.json", "r") as f:
    labels = json.load(f)

REFERENCE_PATH = "D:/Projects/database/LIVE-SJTU_AVQA/Reference/"
DISTORTED_PATH = "D:/Projects/database/LIVE-SJTU_AVQA/Distorted/"

predicts = []
ground_truth = []
for info in tqdm(labels.values()):
    ref_video_path = REFERENCE_PATH + info["ref_video"]
    ref_audio_path = REFERENCE_PATH + info["ref_audio"]
    dis_video_path = DISTORTED_PATH + info["dis_video"]
    dis_audio_path = DISTORTED_PATH + info["dis_audio"]
    ref_video = read_video(ref_video_path)
    ref_audio = read_audio(ref_audio_path)
    dis_video = read_video(dis_video_path)
    dis_audio = read_audio(dis_audio_path)
    score = AVGMSD(ref_video, dis_video, ref_audio, dis_audio)
    predicts.append(score)
    ground_truth.append(info["MOS"])

print("SRCC:", spearmanr(predicts, ground_truth))
