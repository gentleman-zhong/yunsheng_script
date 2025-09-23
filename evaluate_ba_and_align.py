import json
from pyproj import Proj, transform, CRS, Transformer
import numpy as np

def read_shots_info(reconstruction_file):
    """
    读取 reconstruction.json 文件中的 shots 信息
    返回一个 dict，key 为图片名，value 为 {rotation, translation, gps_position}
    """
    with open(reconstruction_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots_info = {}
    for rec in data:  # reconstruction 文件是一个 list
        if "shots" not in rec:
            continue
        for shot_name, shot_data in rec["shots"].items():
            if shot_name not in shots_info:
                shots_info[shot_name] = {
                    "rotation": shot_data.get("rotation", []),
                    "translation": shot_data.get("translation", []),
                    "gps_position": shot_data.get("gps_position", [])
                }
            else:
                print(f"Warning: shot {shot_name} already exists in shots_info.")
    return shots_info

