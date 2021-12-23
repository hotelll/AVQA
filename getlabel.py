import xlrd
import json
import random
from tqdm import tqdm

label_path = "data/LIVE-SJTU_AVQA/MOS.xlsx"

train_ratio = 0.8

wb = xlrd.open_workbook(label_path)

sheet1 = wb.sheet_by_name("allNames")
sheet2 = wb.sheet_by_name("refNames")
sheet3 = wb.sheet_by_name("disNames")
sheet4 = wb.sheet_by_name("MOS")
sheet5 = wb.sheet_by_name("frameNum")
sheet6 = wb.sheet_by_name("frameRate")

sample_num = sheet1.nrows

labels = []

for j in range(sample_num):
    sample_info = {}
    # sheet2
    sample_info["ref_video"] = sheet2.cell_value(j, 0)
    sample_info["ref_audio"] = sheet2.cell_value(j, 1)
    # sheet3
    sample_info["dis_video"] = sheet3.cell_value(j, 0)
    sample_info["dis_audio"] = sheet3.cell_value(j, 1)
    # sheet4
    sample_info["MOS"] = sheet4.cell_value(j, 0)
    # sheet1
    sample_info["compress"] = sheet1.cell_value(j, 1)
    sample_info["kbps"] = sheet1.cell_value(j, 2)
    # sheet5
    sample_info["frame_num"] = sheet5.cell_value(j, 0)
    # sheet6
    sample_info["frame_rate"] = sheet6.cell_value(j, 0)

    labels.append(sample_info)

random.shuffle(labels)
train_num = int(len(labels) * train_ratio)
train_set = labels[: train_num]
test_set =  labels[train_num: ]

train_dict = {}
test_dict = {}

for id, info in enumerate(train_set):
    train_dict[id] = info

for id, info in enumerate(test_set):
    test_dict[id] = info

train_dict = json.dumps(train_dict)
test_dict = json.dumps(test_dict)

f = open("train.json", "w")
f.write(train_dict)
f.close()

f = open("test.json", "w")
f.write(test_dict)
f.close()