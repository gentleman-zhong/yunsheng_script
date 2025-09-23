import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from pyproj import CRS
from pyproj import Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
from scipy.spatial.transform import Rotation as R
import sys
import argparse
import debugpy
import os
import csv
import struct
import glob
from typing import Tuple, Optional
import piexif
from PIL import Image
import open3d as o3d
from plyfile import PlyData, PlyElement
import trimesh

def project_point(
    X_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: Tuple[float, float, float, float, float]
) -> Optional[Tuple[int, int]]:
    """
    将世界坐标系下的三维点投影到二维图像平面上，考虑径向和切向畸变。
    
    参数:
        X_world: 世界坐标系中的3D点，形状为 (3,)。
        R: 相机的旋转矩阵，形状为 (3, 3)。
        t: 相机的平移向量，形状为 (3, 1) 或 (3,)。
        K: 相机内参矩阵，形状为 (3, 3)。
        dist: 相机畸变参数，格式为 (k1, k2, p1, p2, k3)。
    
    返回:
        (u, v): 投影到图像平面上的像素坐标，整数形式。
                如果点在相机后方则返回 None。
    """
    # 坐标转换：世界坐标 -> 相机坐标
    X_cam = R @ X_world.reshape(3, 1) + t.reshape(3, 1)
    X_cam = X_cam.flatten()

    if X_cam[2] <= 0:
        return None  # 点在相机后方，无法投影

    # 归一化相机坐标
    x_n = X_cam[0] / X_cam[2]
    y_n = X_cam[1] / X_cam[2]

    # 解包畸变系数
    k1, k2, p1, p2, k3 = dist

    r2 = x_n**2 + y_n**2
    r4 = r2**2
    r6 = r2**3

    # 径向畸变与切向畸变
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_tangential = 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
    y_tangential = 2 * p2 * x_n * y_n + p1 * (r2 + 2 * y_n**2)

    # 加入畸变
    x_d = x_n * radial + x_tangential
    y_d = y_n * radial + y_tangential

    # 使用内参矩阵将归一化坐标转换为像素坐标
    u = K[0, 0] * x_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]

    return int(round(u)), int(round(v))


def visualize_feature_tracks(
    images_path,
    csv_path,
    output_dir,
    image_size=(5280, 3956),
    max_rows=None,
    point_radius=20,
    point_color=(0, 0, 255),
    thickness=3
):
    """
    将特征点按照 track_id 分组绘制在图像上。

    参数：
        images_path (str): 原始图像所在路径。
        csv_path (str): 包含图像名、特征点坐标和 track_id 的 CSV 文件路径。
        output_dir (str): 绘制结果图像的输出目录。
        image_size (tuple): 图像尺寸，格式为 (width, height)。
        max_rows (int or None): 限制读取 CSV 的行数（调试用）。
        point_radius (int): 圆点半径。
        point_color (tuple): 圆点颜色（BGR 格式）。
        thickness (int): 圆点边缘厚度。
    """
    os.makedirs(output_dir, exist_ok=True)
    width, height = image_size

    try:
        df = pd.read_csv(csv_path, sep='\t', nrows=max_rows)
    except Exception as e:
        print(f"❌ 读取 CSV 文件失败: {e}")
        return

    if not {'image_name', 'x', 'y', 'track_id'}.issubset(df.columns):
        print("❌ CSV 文件缺少必要字段：'image_name', 'x', 'y', 'track_id'")
        return

    grouped_tracks = df.groupby('track_id')
    image_cache = {}

    for track_id, group in grouped_tracks:
        for _, row in group.iterrows():
            image_name = row['image_name']
            u, v = float(row['x']), float(row['y'])

            # OpenSfM 输出的 u,v 是 [-0.5, 0.5] 范围
            x = int((width - 1) / 2 + u * width)
            y = int((height - 1) / 2 + v * width)

            img_path = os.path.join(images_path, image_name)
            if image_name not in image_cache:
                if not os.path.exists(img_path):
                    print(f"⚠️ 图像不存在: {img_path}")
                    continue
                image_cache[image_name] = cv2.imread(img_path)

            cv2.circle(image_cache[image_name], (x, y), point_radius, point_color, thickness)

    # 保存所有带有标记的图像
    for name, img in image_cache.items():
        save_path = os.path.join(output_dir, name)
        cv2.imwrite(save_path, img)

    print(f"✅ 特征点可视化完成，保存目录：{output_dir}")

 
 

    def clearExifInfo(photoAddress):
        image = Image.open(photoAddress)
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        image_without_exif.save(photoAddress)
    countNums = 0    
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".JPG") or name.endswith(".jpg"):
                photoAddress = os.path.join(root,name)
 
                clearExifInfo(photoAddress)
 
                #print(photoAddress)
                countNums += 1
                print("已处理{0}目录中，总第{1}张图像".format(root, countNums ))


# 测试opensfm的相机参数
# from opensfm.dataset import DataSet

def camera_to_colmap_params(camera):
    w = camera.width
    h = camera.height
    normalizer = max(w, h)
    print(normalizer)
    f = camera.focal * normalizer
    if camera.projection_type in ("perspective", "fisheye"):
        k1 = camera.k1
        k2 = camera.k2
        cx = w * 0.5
        cy = h * 0.5
        return f, cx, cy, k1, k2
    elif camera.projection_type == "brown":
        fy = f * camera.aspect_ratio
        c_x = w * 0.5 + normalizer * camera.principal_point[0]
        c_y = h * 0.5 + normalizer * camera.principal_point[1]
        k1 = camera.k1
        k2 = camera.k2
        k3 = camera.k3
        p1 = camera.p1
        p2 = camera.p2
        return f, fy, c_x, c_y, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0
    elif camera.projection_type == "fisheye_opencv":
        fy = f * camera.aspect_ratio
        cx = w * 0.5 + camera.principal_point[0]
        cy = h * 0.5 + camera.principal_point[1]
        k1 = camera.k1
        k2 = camera.k2
        k3 = camera.k3
        k4 = camera.k4
        return f, fy, cx, cy, k1, k2, k3, k4
    else:
        raise ValueError("Can't convert {camera.projection_type} to COLMAP")

def get_odm_camera_info(data_path):
    data = DataSet(data_path)
    reconstructions = data.load_reconstruction()
    cameras = {}
    for reconstruction in reconstructions:
        for camera_id, camera in reconstruction.cameras.items():
            cameras[camera_id] = camera

    camera_info = {}
    for camera_id, camera in cameras.items():
        w = camera.width
        h = camera.height
        params = camera_to_colmap_params(camera)
        camera_info[camera_id] = {
            'width': w,
            'height': h,
            'params': params
        }
    return camera_info

def remove_metadata_from_folder(folder_path):
    """
    删除文件夹中所有图像的元数据（EXIF）
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        # 清除 EXIF 信息
                        img.info.pop("exif", None)
                        img.save(image_path)
                    print(f"已处理: {image_path}")
                except Exception as e:
                    print(f"处理失败 {image_path}: {e}")

def format_gps_info(gps_ifd):
    def decode(coord):
        if not isinstance(coord, tuple) or len(coord) != 3:
            return "?"
        d, m, s = coord
        try:
            return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600
        except:
            return "?"
    if not gps_ifd:
        return "No GPS info"
    lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode(errors='ignore')
    lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode(errors='ignore')
    if lat and lon:
        lat_val = decode(lat)
        lon_val = decode(lon)
        return f"Lat: {lat_val:.6f}° {lat_ref}, Lon: {lon_val:.6f}° {lon_ref}"
    return "GPS tags present but incomplete"

def sanitize_exif(exif_dict):
    for ifd_name in exif_dict:
        if not isinstance(exif_dict[ifd_name], dict):
            continue
        for tag in list(exif_dict[ifd_name].keys()):
            value = exif_dict[ifd_name][tag]
            expected_type = piexif.TAGS[ifd_name].get(tag, {}).get("type")
            if expected_type in [2, 7]:  # ASCII or UNDEFINED
                if isinstance(value, int):
                    print(f"  ⚠️  Removing invalid tag {tag} from {ifd_name}")
                    del exif_dict[ifd_name][tag]
    return exif_dict

def remove_gps_without_recompression(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(file_path)
            exif_bytes = img.info.get("exif", b"")
            if not exif_bytes:
                print(f"{filename}: No EXIF data found.\n")
                continue

            exif_dict = piexif.load(exif_bytes)
            gps_info = exif_dict.get("GPS", {})
            if gps_info:
                print(f"{filename}: {format_gps_info(gps_info)}")
                exif_dict["GPS"] = {}
                exif_dict = sanitize_exif(exif_dict)
                new_exif_bytes = piexif.dump(exif_dict)
                piexif.insert(new_exif_bytes, file_path)  # ✔ inject EXIF directly, no re-save
                print(f"  ✅ GPS removed without re-encoding\n")
            else:
                print(f"{filename}: No GPS data.\n")

        except Exception as e:
            print(f"{filename}: Failed with error: {e}\n")


def convert_ply2denseply(input_path, output_path):
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_path)
    # pcd = pcd.random_down_sample(sampling_ratio)

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros_like(points)
    views = np.zeros((points.shape[0], 1), dtype=np.uint8)

    # 准备 structured array
    data = np.empty(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('views', 'u1')
    ])

    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['red'] = colors[:, 0]
    data['green'] = colors[:, 1]
    data['blue'] = colors[:, 2]
    data['nx'] = normals[:, 0]
    data['ny'] = normals[:, 1]
    data['nz'] = normals[:, 2]
    data['views'] = views[:, 0]

    # 写入 PLY（二进制）
    ply_el = PlyElement.describe(data, 'vertex')
    ply_data = PlyData([ply_el], text=False)

    # 将 header 中的 'float' 替换为 'float32'
    import io
    buf = io.BytesIO()
    ply_data.write(buf)
    content = buf.getvalue()

    # 替换 header 中的 float -> float32
    header_end = content.find(b'end_header\n') + len(b'end_header\n')
    header = content[:header_end].decode('utf-8')
    header = header.replace('property float ', 'property float32 ')
    header = header.replace('property uchar ', 'property uint8 ')
    new_content = header.encode('utf-8') + content[header_end:]

    # 写入最终文件
    with open(output_path, 'wb') as f:
        f.write(new_content)

    print(f"Saved corrected PLY with explicit float32: {output_path}")

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    max_x, max_y, max_z = points.max(axis=0)
    print("点云数量:", points.shape[0])
    print("前5个点坐标:\n", points[:5])

def read_mesh(file_path):
    scene = trimesh.load(file_path, process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    else:
        mesh = scene
    vertices = np.array(mesh.vertices)
    max_x, max_y, max_z = vertices.max(axis=0)
    print("顶点数量:", vertices.shape[0])
    print("前5个顶点:\n", vertices[:5])

read_ply("/home/zhangzhong/experiment/Optimize_ODM_MVS_Mesh/test_redolution_level/dongman/opensfm/undistorted/openmvs/resolution_level_results/3/scene_dense_dense_filtered.ply")
read_mesh("/home/zhangzhong/experiment/Optimize_ODM_MVS_Mesh/test_redolution_level/dongman/opensfm/undistorted/openmvs/resolution_level_results/3/odm_texturing/odm_textured_model_geo.obj")
# convert_ply2denseply(input_path="/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_dongman/opensfm/undistorted/openmvs/test_enu.ply", output_path="/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_dongman/opensfm/undistorted/openmvs/scene_dense_dense_filtered.ply")
# convert_ply2meshply