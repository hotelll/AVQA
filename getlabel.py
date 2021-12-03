import xlrd
import json

label_path = "D:/Projects/database/LIVE-SJTU_AVQA/MOS.xlsx"

wb = xlrd.open_workbook(label_path)

sheet1 = wb.sheet_by_name("allNames")
sheet2 = wb.sheet_by_name("refNames")
sheet3 = wb.sheet_by_name("disNames")
sheet4 = wb.sheet_by_name("MOS")
sheet5 = wb.sheet_by_name("frameNum")
sheet6 = wb.sheet_by_name("frameRate")

sample_num = sheet1.nrows

dict_result = {}

for j in range(sample_num):
    sample_info = {}
    # sheet2
    sample_info["ref_video"] = sheet2.cell_value(j, 0)
    sample_info["ref_audio"] = sheet2.cell_value(j, 1)
    # sheet3
    sample_info["dis_video"] = sheet3.cell_value(j, 0)
    sample_info["dis_audio"] = sheet3.cell_value(j, 1)
    # sheet4
    sample_info["MOS"]       = sheet4.cell_value(j, 0)
    # sheet1
    sample_info["compress"] = sheet1.cell_value(j, 1)
    sample_info["kbps"] = sheet1.cell_value(j, 2)
    # sheet5
    sample_info["frame_num"] = sheet5.cell_value(j, 0)
    # sheet6
    sample_info["frame_rate"] = sheet6.cell_value(j, 0)

    dict_result[j] = sample_info

str_result = json.dumps(dict_result)
f = open("label.json", "w")
f.write(str_result)
f.close()