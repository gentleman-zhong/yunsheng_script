import argparse
import csv
import os
from opensfm.dataset import DataSet
from opensfm import pymap
import shutil

def convert_binary_to_csv_tracks(data_path: str):
    """Convert OpenSfM tracks to CSV format."""
    data = DataSet(data_path)
    tracks_manager = data.load_tracks_manager()
    output_path = os.path.join(data_path, "converted_tracks.csv")

    with open(output_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')  # 使用 tab 分隔符
        writer.writerow(["image_name", "track_id", "feature_id", "x", "y", "scale", "r", "g", "b", "-1", "-1"])
        
        for track_id in tracks_manager.get_track_ids():
            observations = tracks_manager.get_track_observations(track_id)
            for image_name, obs in observations.items():
                feature_id = obs.id
                x, y = obs.point[0], obs.point[1]
                scale = obs.scale
                r, g, b = obs.color
                writer.writerow([
                    image_name,
                    track_id,
                    feature_id,
                    x,
                    y,
                    scale,
                    r,
                    g,
                    b,
                    -1,
                    -1
                ])
    print(f"[✓] Successfully Convert CSV to: {output_path}")

def read_custom_csv(file_path):
    records = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('image'):
                continue  # 跳过标题行
            parts = line.strip().split()
            if len(parts) != 11:
                continue  # 跳过格式错误的行

            image = parts[0]
            track_id = int(parts[1])
            feature_id = int(parts[2])
            x = float(parts[3])
            y = float(parts[4])
            scale = float(parts[5])
            r = int(parts[6])
            g = int(parts[7])
            b = int(parts[8])
            # 后面两个字段 -1 -1 可以忽略，或按需使用

            records.append({
                'image': image,
                'track_id': track_id,
                'feature_id': feature_id,
                'x': x,
                'y': y,
                'scale': scale,
                'color': (r, g, b)
            })

    return records

def convert_csv_to_binary_tracks(data_path: str):
    # help(pymap.Observation)
    # parent_path = os.path.dirname(os.path.dirname(os.path.dirname(data_path)))
    parent_path = "/home/zhangzhong/experiment/one_km/099-4-7地块-routeid-76386"
    data = DataSet(parent_path)
    # print(data.tracks_exists())
    tracks_records = read_custom_csv(os.path.join(data_path, "converted_tracks.csv"))
    tracks_manager = pymap.TracksManager()
    # help(tracks_manager.add_observation)
    for record in tracks_records:
        track_id_str = str(record['track_id'])        
        observation = pymap.Observation(
            x = float(record['x']),          # x: float
            y = float(record['y']),          # y: float
            s = 0.1,                         # s: float (这里暂时使用 0 代替)
            r = int(record['color'][0]),     # r: int
            g =int(record['color'][1]),     # g: int
            b = int(record['color'][2]),     # b: int
            feature = int(record['feature_id']),   # feature: int (假设这里传递的是 feature_id)
            segmentation = -1,                          # segmentation: int, 默认值为 -1
            instance = -1                           # instance: int, 默认值为 -1
        )
        tracks_manager.add_observation(str(record['image']),track_id_str, observation)    
    data.save_tracks_manager(tracks_manager = tracks_manager)
# 获取原文件路径
    original_file_path = os.path.join(parent_path, 'tracks.csv')
    # 目标文件路径
    destination_file_path = os.path.join(data_path, 'tracks.csv')
    # 移动文件
    shutil.move(original_file_path, destination_file_path)
    print("successfully saved tracks data")

def main():
    parser = argparse.ArgumentParser(description="Convert OpenSfM track data to CSV.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the OpenSfM project folder.")
    parser.add_argument("--to_binary", action="store_true", help="which format to convert to, binary or csv.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

    if args.to_binary:
        convert_csv_to_binary_tracks(args.data_path)
    else:
        convert_binary_to_csv_tracks(args.data_path)
    


if __name__ == "__main__":
    main()