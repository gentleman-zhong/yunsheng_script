import argparse
import csv
import os
from opensfm.dataset import DataSet
from opensfm import pymap
import shutil
import json
import cv2

def get_tracks_and_pixel_num(data_path: str,visible_ratio: float = 0.6):
    data = DataSet(data_path)
    tracks_manager = data.load_tracks_manager()
    point_cloud_sparese_num = len(tracks_manager.get_track_ids())
    point_cloud_pixel_num = 0
    for track_id in tracks_manager.get_track_ids():
        observations = tracks_manager.get_track_observations(track_id)
        l = len(observations)
        point_cloud_pixel_num += l


    exif = data.load_exif(list(data.image_files.keys())[0])
    width = exif['width']
    height = exif['height']
    num = len(list(data.image_files.keys()))
    max_mvs_point_num = int(width * height * num * visible_ratio * point_cloud_sparese_num / point_cloud_pixel_num)
    return width, height, num, max_mvs_point_num


def main():
    parser = argparse.ArgumentParser(description="Convert OpenSfM track data to CSV.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the OpenSfM project folder.")
    args = parser.parse_args()

    width, height, num, max_mvs_point_num= get_tracks_and_pixel_num(args.data_path)
    print("width:", width)
    print("height:", height)
    print("num:", num)
    print("max_mvs_point_num:", max_mvs_point_num)

 

if __name__ == "__main__":
    main()