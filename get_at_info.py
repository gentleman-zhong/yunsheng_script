import argparse
import os
import json
from typing import Tuple, Dict, Any, Optional
from las_2_3dtiles import convert_and_write_las, convert_las_to_3dtiles, parse_coords_file, export_shots_json
from pathlib import Path
import xml.etree.ElementTree as ET
from pyproj.database import query_utm_crs_info
import numpy as np
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.transformer import Transformer


def read_reconstruction_info(reconstruction_file):
    position_num = 0
    rotation_num = 0
    photos_num = 0
    pixel_num = 0
    tracks_points_num = 0

    with open(reconstruction_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_points = []
    all_cameras = {}
    all_shots = {}
    for idx,  rec_info in enumerate(data):
        if idx == 0:
              main_components_num = len(rec_info["shots"])
        cameras_info = rec_info["cameras"]
        shots_info = rec_info["shots"]
        points_info = rec_info["points"]

        for camera_name,cmaera_data in cameras_info.items():
                if camera_name not in all_cameras:
                    all_cameras[camera_name] = cmaera_data

        for shot_name, shot_data in shots_info.items():
                if shot_name not in all_shots:
                    all_shots[shot_name] = shot_data
                    photos_num += 1
                    if shot_data.get("rotation", []):
                        rotation_num += 1
                    if shot_data.get("translation", []):
                        position_num += 1
                    if shot_data.get("camera", []):
                        local_camera_name = shot_data.get("camera", [])
                        # 先按照百万像素换算
                        pixel_num += cameras_info[local_camera_name]["width"] * cameras_info[local_camera_name]["height"] / 1000000
        
        tracks_points_num += len(points_info)
        all_points.extend(points_info.values())
    photogroup_num = len(all_cameras)

    return (photos_num, photogroup_num, round(pixel_num / 1000, 1), main_components_num, position_num, rotation_num, tracks_points_num), all_points, all_cameras, all_shots

def read_gcp_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    all_gcp_id = []
    # 解析后续行
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # 跳过异常行
        gcp_id = parts[6] 
        if gcp_id not in  all_gcp_id:
             all_gcp_id.append(gcp_id)                       
    return len(all_gcp_id)

def build_summary_json(rec_info: Tuple, gcp_num: int = 0, user_tie_points_num: int = 0) -> Dict[str, Any]:
    # 如果 rec_info 是 tuple/list，就按位置取值；不足的用 None 填充
    defaults = [None] * 7
    if isinstance(rec_info, (list, tuple)):
        for i, v in enumerate(rec_info):
            if i < len(defaults):
                defaults[i] = v
    else:
        raise TypeError("rec_info should be a tuple or list")

    summary = {
        "photos_num": defaults[0],
        "photogroup_num": defaults[1],
        "total_gigapixels": defaults[2],        # 已是 round(pixel_num/1000, 1)
        "main_components_shots": defaults[3],
        "position_num": defaults[4],
        "rotation_num": defaults[5],
        "gcp_num": int(gcp_num),
        "user_tie_points_num": int(user_tie_points_num),
        "tracks_points_num": defaults[6],
    }

    return summary

def read_reference_lla(reference_file):
    """
    读取 reference_lla.json 文件中的参考点信息
    返回 (lat, lon)
    """
    import json
    with open(reference_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def convert_origin(_origin):
  # 纬度经度海拔
  ori_gps = np.array([_origin['latitude'], _origin['longitude'], _origin['altitude']])
  crs_wgs84 = CRS.from_epsg(4326)
  aoi = AreaOfInterest(west_lon_degree = ori_gps[1], south_lat_degree = ori_gps[0], east_lon_degree = ori_gps[1], north_lat_degree = ori_gps[0])
  utm_crs_list = query_utm_crs_info(datum_name = "WGS 84",area_of_interest = aoi)
  crs_utm = CRS.from_epsg(utm_crs_list[0].code)
  transformer = Transformer.from_crs(crs_wgs84, crs_utm)
  ori_utm = transformer.transform(ori_gps[0], ori_gps[1])
  ori_utm = np.array([ori_utm[0], ori_utm[1], ori_gps[2]])
  
  return ori_utm

def get_metadata_xml(ori_utm, metadeata_xml_path, espg_number):
  # 构建 XML 结构
  root = ET.Element("ModelMetadata", version="1")

  srs = ET.SubElement(root, "SRS")
  srs.text = f"EPSG:{espg_number}"

  srs_origin = ET.SubElement(root, "SRSOrigin")
  srs_origin.text = f"{float(ori_utm[0])},{float(ori_utm[1])},{float(ori_utm[2])}"

  texture = ET.SubElement(root, "Texture")
  color_source = ET.SubElement(texture, "ColorSource")
  color_source.text = "Visible"

  # 写入 XML 文件
  tree = ET.ElementTree(root)
  tree.write(metadeata_xml_path, encoding='utf-8', xml_declaration=True)

def convert_reconstruction_info_to_cessium(opensfm_path, all_points, all_cameras, all_shots):
    # Step 0: 创建输出文件夹
    output_dir = os.path.join(opensfm_path, "visualization_dir")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Step 1: 读取coords文件
    proj_file = os.path.join(os.path.dirname(opensfm_path), "odm_georeferencing/coords.txt")
    if not os.path.exists(proj_file):
        espg_number = None
        offset = None
    else:
        espg_offset_dirct = parse_coords_file(proj_file)
        espg_number = espg_offset_dirct["epsg"]
        offset = espg_offset_dirct["offset"]

    # Step 2: 先写 las 文件
    las_path = Path(os.path.join(output_dir, "points.las"))
    convert_and_write_las(all_points, offset, out_path=las_path)

    # Step 3: 用 py3dtiles 转换 las -> 3dtiles
    convert_las_to_3dtiles(output_dir, "py3dtiles", espg_number)

    # Step 4: 相机位置和相机存储，UTM坐标，GPS坐标
    export_shots_json(all_cameras, all_shots, offset, output_dir, espg_number)

    # Step 5: 存储坐标原点信息
    metadeata_xml_path = os.path.join(output_dir, "metadata.xml")
    reference_file = os.path.join(opensfm_path, "reference_lla.json")
    if os.path.exists(reference_file):
        origin = read_reference_lla(reference_file)
        ori_utm = convert_origin(origin)
    else:
        ori_utm = [0, 0, 0]
    
    if espg_number is not None:
        get_metadata_xml(ori_utm, metadeata_xml_path, espg_number)
    else:
        get_metadata_xml(ori_utm, metadeata_xml_path, 32650)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert OpenSfM track data to CSV.")
    parser.add_argument("--opensfm_path", type=str, required=True, help="Path to the OpenSfM project folder.")
    args = parser.parse_args()

    reconstruction_file = os.path.join(args.opensfm_path, "reconstruction.json")
    gcp_file = os.path.join(args.opensfm_path, "gcp_list.txt")

    if os.path.exists(gcp_file):
        gcp_num = read_gcp_file(gcp_file)
    else:
        gcp_num = 0

    rec_info, all_points, all_cameras, all_shots = read_reconstruction_info(reconstruction_file)
    # 最终需要的重建结果总结
    summary = build_summary_json(rec_info, gcp_num)
    # 将结果转换成能够可视化需要的的格式
    convert_reconstruction_info_to_cessium(args.opensfm_path, all_points, all_cameras, all_shots)
    
    # 把summary写成文件
    with open(os.path.join(args.opensfm_path,  "visualization_dir", "summary.json"), "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)
