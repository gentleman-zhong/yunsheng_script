import json
from pyproj import Proj, transform, CRS, Transformer
import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

def read_reference_lla(reference_file):
    """
    读取 reference_lla.json 文件中的参考点信息
    返回 (lat, lon)
    """
    import json
    with open(reference_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data["latitude"], data["longitude"])

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

def get_camera_position(rotation, translation):
    r_matrix = R.from_rotvec(rotation).as_matrix()
    t = np.array(translation)
    # 相机在世界坐标系中的位置
    cam_center = -r_matrix.T @ t
    return cam_center

def xml_to_gcp(xml_file, gcp_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 找到投影信息
    srs = root.find(".//SRS/Name")
    projection = ""
    if srs is not None:
        name = srs.text
        if "UTM zone" in name:
            # 简单转换成 proj4 格式
            zone = name.split("zone")[1].split()[0]
            projection = f"+proj=utm +zone={zone[0:-1]} +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        else:
            projection = name
    else:
        projection = "+proj=longlat +datum=WGS84 +no_defs"

    lines = [projection]

    # 遍历 TiePoints
    for idx, tp in enumerate(root.findall(".//TiePoint")):
        gcp_name = tp.find("Name").text.strip()
        pos = tp.find("Position")
        geo_x = pos.find("x").text
        geo_y = pos.find("y").text
        geo_z = pos.find("z").text

        for meas in tp.findall("Measurement"):
            im_x = meas.find("x").text
            im_y = meas.find("y").text
            image_path = meas.find("ImagePath").text
            image_name = os.path.basename(image_path)

            line = f"{geo_x} {geo_y} {geo_z} {im_x} {im_y} {image_name} gcp{idx}"
            lines.append(line)

    # 写出文件
    with open(gcp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 已成功转换: {xml_file} → {gcp_file}")

def utm_to_enu(x, y, z, lat0, lon0, alt0, utm_zone=50, northern=True):
    """
    将 WGS84 UTM 坐标转换为 ENU 坐标系
    
    参数:
        x, y, z: UTM 坐标 (m)
        lat0, lon0, alt0: 原点经纬度 (degrees, meters)
        utm_zone: UTM 分区 (默认 50)
        northern: True=北半球, False=南半球
    返回:
        (e, n, u): ENU 坐标 (m)
    """

    # 1. UTM -> WGS84 经纬度
    crs_utm = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs{' +north' if northern else ' +south'}")
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)

    # 2. 经纬度 -> ECEF
    def lla_to_ecef(lat, lon, alt):
        # WGS84 参数
        a = 6378137.0  # 长半轴
        e = 8.1819190842622e-2  # 偏心率

        lat, lon = np.radians(lat), np.radians(lon)
        N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)

        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (N * (1 - e**2) + alt) * np.sin(lat)
        return np.array([X, Y, Z])

    # 控制点
    ecef = lla_to_ecef(lat, lon, z)
    # 原点
    ecef_ref = lla_to_ecef(lat0, lon0, alt0)

    # 3. ECEF -> ENU
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    R = np.array([
        [-np.sin(lon0_rad),              np.cos(lon0_rad),             0],
        [-np.sin(lat0_rad)*np.cos(lon0_rad), -np.sin(lat0_rad)*np.sin(lon0_rad), np.cos(lat0_rad)],
        [ np.cos(lat0_rad)*np.cos(lon0_rad),  np.cos(lat0_rad)*np.sin(lon0_rad), np.sin(lat0_rad)]
    ])

    enu = R @ (ecef - ecef_ref)
    return tuple(enu)


def main1():
    parser = argparse.ArgumentParser(description="get gps residual")
    parser.add_argument("--opensfm_dir", type=str, required=True, help="Path to the OpenSfM project folder.")
    args = parser.parse_args()
    reconstruction_file = os.path.join(args.opensfm_dir, "reconstruction.json")
    shots_info = read_shots_info(reconstruction_file)
    
    
    gps_distance_dict = {}
    for shot_name, shot_data in shots_info.items():
        if len(shot_data["gps_position"]) == 3:
            enu_gps = shot_data["gps_position"]
            cam_center_enu = get_camera_position(shot_data["rotation"], shot_data["translation"])
            gps_distance_dict[shot_name] = np.linalg.norm(enu_gps - cam_center_enu)
        else:
            print(f"Warning: shot {shot_name} has no GPS position.")

    errors = np.array(list(gps_distance_dict.values()))

    gps_residual = {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean": np.mean(errors),
        "max": np.max(errors),
        "per_shot": gps_distance_dict
    }

    with open(os.path.join(args.opensfm_dir, "gps_residual.json"), "w", encoding="utf-8") as f:
        json.dump(gps_residual, f, indent=4)


def main2():
    parser = argparse.ArgumentParser(description="get gcp residual")
    parser.add_argument("--opensfm_dir", type=str, required=True, help="Path to the OpenSfM project folder.")
    args = parser.parse_args()

    gcp_file = os.path.join(args.opensfm_dir, "gcp_list.txt")
    



if __name__ == "__main__":
    main2()