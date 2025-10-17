import numpy as np
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from pyproj import CRS
from scipy.spatial.transform import Rotation as R
from pyproj import Transformer
import json
import laspy

def parse_coords_file(coords_file: str) -> Dict[str, Optional[object]]:
    coords_file = Path(coords_file)
    if not coords_file.exists(): 
        return {"epsg": None, "offset": None}
    with coords_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        return {"epsg": None, "offset": None}
    proj_info = lines[0]
    try:
        if "UTM" in proj_info:
            parts = proj_info.split() 
            zone = None 
            hemi = None 
            for i, p in enumerate(parts): 
                 if p.upper() == "UTM" and i + 1 < len(parts): 
                    zone = int(parts[i + 1][:-1]) # 提取 zone 
                    hemi = parts[i + 1][-1].upper() 
            if hemi == "N": 
                epsg = 32600 + zone 
            elif hemi == "S": 
                epsg = 32700 + zone 
            else: epsg = None 
        else: epsg = None 
    except Exception: 
        epsg = None
    # 解析第二行 offset 
    try: 
        x_str, y_str = lines[1].split() 
        offset = (float(x_str), float(y_str)) 
    except Exception: 
        offset = None 

    return {"epsg": epsg, "offset": offset}

def get_las_header_attrs(point_format=7, version="1.4"):
    """
    根据 point_format 和 version 获取 las 文件的 header 属性
    说明文档：https://laspy.readthedocs.io/en/latest/intro.html#point-records
    Args:
        point_format: 点格式
        version: 版本

    Returns:

    """
    dimensions = []
    header = laspy.LasHeader(point_format=point_format, version=version)  # 7 支持rgb
    for dim in header.point_format.dimensions:
        dimensions.append(dim.name)
    return dimensions

def write_las_fit(out_file, xyz, rgb=None, attrs=None):
    if attrs is None:
        attrs = {}


    point_format = attrs.pop("point_format") if "point_format" in attrs else 7  # 认为 7 支持rgb
    version = attrs.pop("version") if "version" in attrs else "1.4" # 默认 1.4
    header = laspy.LasHeader(point_format=point_format, version=version)

    ''' 自动计算 scales 和 offsets，确保坐标精度无损 '''
    header.scale = attrs.pop("scales") if "scales" in attrs else [0.01, 0.01, 0.01]  # 0.001 是毫米精度
    # 如果保存时与原las文件的offsets不同，可能会出现偏移（偏移精度 < scales精度限制）
    header.offset = attrs.pop("offsets") if "offsets" in attrs else np.floor(np.min(xyz, axis=0))

    ''' 初始化一些需要保存的属性值。 '''
    extra_attr = []  # 如果是额外属性，添加到 header 中, 后续赋值
    # 获取当前版本的保留字段（与 point_format 和 version 有关）
    all_reserved_fields = get_las_header_attrs(point_format, version)
    for attr, value in attrs.items():
        if attr not in all_reserved_fields:   # 添加额外属性，在 las 初始化后赋值，如 label, pred
            header.add_extra_dim(laspy.ExtraBytesParams(name=attr, type=np.float32))
            extra_attr.append(attr)

    ''' 创建 las 文件对象 '''
    las = laspy.LasData(header)

    # 添加xyz坐标
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    ''' 添加颜色信息 '''
    # 如果RGB全是0, 则不存储颜色。如果是归一化的颜色，则需要乘以 65535，转为 uint16
    if (rgb is not None) and (np.max(rgb) > 0):
        if np.max(rgb) <= 1:
            rgb = (rgb * 65535).astype(np.uint16)  # 65535 = 2^16 - 1, las存储颜色是16位无符号整型
        elif np.max(rgb) <= 255:
            rgb = (rgb / 255 * 65535).astype(np.uint16)
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]

    ''' 设置固有属性 '''
    for attr, value in attrs.items():
        if attr in all_reserved_fields:  # 如果是保留字段，则不添加额外属性。如 X, Y, Z, Red, Green, Blue
            if value.ndim == 2 and value.shape[1] == 1:
                value = value.flatten()
            las[attr] = value

    ''' 设置额外属性 '''
    for attr in extra_attr:
        # 当 value 是 n * 1 的 ndarray 时，转换为 1 维数组
        value = attrs[attr]
        if value.ndim == 2 and value.shape[1] == 1:
            value = value.flatten()
        las[attr] = value

    ''' 写入文件 '''
    las.write(out_file)

def convert_and_write_las(all_points, offset=None, out_path: Path = None):
    """
    将点云数据写入 LAS 文件，并在写入前对坐标进行 XY 平移。

    参数:
        all_points (list[dict]): 包含点的列表，每个点是一个字典，至少包含 'coordinates' 和 'color' 键。
        offset (tuple[float, float], optional): XY 平移量，例如 (dx, dy)。默认为 None，不进行平移。
        out_path (Path): 输出 LAS 文件路径。
    """
    # 提取坐标和颜色
    coords = np.array([p['coordinates'] for p in all_points], dtype=np.float64)  # (N, 3)
    colors = np.array([p['color'] for p in all_points], dtype=np.float32)       # (N, 3)

    # 应用 XY 平移
    if offset is not None:
        coords[:, 0] += offset[0]  # X 平移
        coords[:, 1] += offset[1]  # Y 平移

    # 写入 LAS 文件
    write_las_fit(out_path, coords, colors)


def which_py3dtiles(user_cmd: Optional[str] = None) -> str:
    """返回 py3dtiles 可执行路径（优先 user_cmd），找不到抛错。"""
    if user_cmd:
        return user_cmd
    p = shutil.which("py3dtiles")
    if p:
        return p
    raise RuntimeError("未找到 py3dtiles，请安装并确保可执行文件在 PATH，或通过参数指定路径。")

def convert_las_to_3dtiles(
    output_dir: str,
    py3dtiles_cmd: Optional[str] = None,
    epsg_number: Optional[int] = None,
    srs_out: str = "4978",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    将指定目录中的 points.las 转换为 3D Tiles，结果输出到 output_dir/3dtiles。

    参数:
      - output_dir (str): 包含 points.las 的文件夹路径。
      - py3dtiles_cmd (str, 可选): py3dtiles 可执行文件路径，若为 None 则默认在 PATH 中查找。
      - epsg_number (int, 可选): 输入点云的 EPSG 编码，若提供则作为 --srs_in 参数。
      - srs_out (str): 输出坐标系 EPSG 编码（默认 4978，即 WGS84 ECEF）。
      - overwrite (bool): 是否添加 --overwrite 参数，覆盖已有输出。

    返回:
      dict: 包含以下键
        - ok (bool): 是否成功执行。
        - out_dir (Path|None): 输出 3D Tiles 文件夹路径，失败时为 None。
        - message (str): 执行输出或错误信息。
    """
    output_dir = Path(output_dir)
    las_path = output_dir / "points.las"
    if not las_path.exists():
        return {"ok": False, "out_dir": None, "message": f"未找到 LAS 文件: {las_path}"}

    out_dir = output_dir / "3dtiles"
    py3dtiles = which_py3dtiles(py3dtiles_cmd)

    cmd = [py3dtiles, "convert", str(las_path), "--out", str(out_dir)]
    if overwrite:
        cmd.append("--overwrite")

    # 输入/输出坐标系
    if epsg_number is not None:
        cmd += ["--srs_in", str(epsg_number), "--srs_out", str(srs_out)]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        ok = (proc.returncode == 0)
        msg = proc.stdout.strip()
        return {
            "ok": ok,
            "out_dir": out_dir if ok else None,
            "message": msg or f"returncode={proc.returncode}",
        }
    except Exception as e:
        return {"ok": False, "out_dir": None, "message": f"Exception: {repr(e)}"}

def get_cessium_shot_info(rotation, translation, offset):
    r_matrix = R.from_rotvec(rotation).as_matrix()
    
    offset = np.array([offset[0], offset[1], 0.0])
    # 更新平移向量
    t_new = translation - r_matrix @ offset
    
    # 构建 4x4 位姿矩阵
    M = np.eye(4)
    M[:3, :3] = r_matrix
    M[:3, 3] = t_new
    M_inv = np.linalg.inv(M)

    position_001 = np.array([0.0, 0.0, 1.0, 1.0])  # 齐次坐标
    position_001_w = M_inv @ position_001  # 相机坐标系→世界坐标系
    position_001_w = position_001_w[0:3]  # 齐次坐标 → 非齐次坐标

    # 从逆矩阵提取旋转和平移
    t_c2w = M_inv[:3, 3]       # 相机坐标系→世界坐标系平移
    return t_c2w.tolist(),position_001_w.tolist()

def utm_to_lonlat(easting, northing, height=None, epsg_number=32650):
    """
    将 UTM/投影坐标 (easting, northing, height) -> WGS84 (lon, lat, height).
    参数:
        easting (float): UTM 东坐标（X）
        northing (float): UTM 北坐标（Y）
        height (float or None): 高度（可为 None）
        epsg_number (int or str): 源投影 EPSG 编号，例如 32650
    返回:
        (lon, lat, height) 三元组，lon/lat 为浮点经纬度（度），height 原样返回（若为 None 则返回 None）
    说明:
        - 使用 pyproj.Transformer.from_crs(..., always_xy=True) 确保输入顺序为 (x, y)。
        - EPSG:326xx 为 WGS84 UTM 北半球（zone xx），EPSG:327xx 为南半球。
    """
    src_crs = f"EPSG:{epsg_number}"
    dst_crs = "EPSG:4326"  # WGS84 lon/lat
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    if height is None:
        return float(lon), float(lat), None
    else:
        return float(lon), float(lat), float(height)
    

def export_shots_json(all_cameras, all_shots, offset, output_dir, epsg_number, start_img_id=10000):
    outpt_path = Path(output_dir) / "features.json"
    
    for cam_name, cam_data in all_cameras.items():
        scale = max(cam_data["width"], cam_data["height"])
        cam_data["focal_x"] *= scale
        cam_data["focal_y"] *= scale
        cam_data["c_x"] = (cam_data["width"] - 1) / 2 + cam_data["c_x"] * scale
        cam_data["c_y"] = (cam_data["height"] - 1) / 2 + cam_data["c_y"] * scale
    
    features = []
    for idx, (shot_name, shot_data) in enumerate(all_shots.items()):
        translation_c2w, end_utm = get_cessium_shot_info(shot_data["rotation"], shot_data["translation"], offset)
        
        start_gps = utm_to_lonlat(translation_c2w[0], translation_c2w[1], height=translation_c2w[2], epsg_number=epsg_number)
        end_gps = utm_to_lonlat(end_utm[0], end_utm[1], height=end_utm[2], epsg_number=epsg_number)

        # ---- 2. 获取 camera 信息（可选） ----
        cam_key = shot_data.get("camera")
        cam_info = all_cameras.get(cam_key, {}) if cam_key is not None else {}
        focal = cam_info.get("focal_x") or cam_info.get("focal")  # 像素为单位的焦距（若你想保留像素）
        width = cam_info.get("width")
        height = cam_info.get("height")

        # 所以这里我们直接把 focal_x (pixel) 填入，或 None：
        focal_out = float(focal) if focal is not None else None

        # ---- 3. capture_time -> 毫秒（若 OpenSfM 是秒） ----
        capture_time = shot_data.get("capture_time")
        if capture_time is None:
            capture_time_ms = None
        else:
            # 如果 capture_time 看起来像秒（例如 1.7e9），则乘 1000
            if capture_time < 1e12:
                capture_time_ms = int(capture_time * 1000)
            else:
                capture_time_ms = int(capture_time)

        # ---- 4. geometry.coordinates: 如果 gps_position 看起来像经纬度则使用，否则 None ----
        geometry_coords = None
        try:
            easting = translation_c2w[0]
            northing = translation_c2w[1]
            z = translation_c2w[2] if len(translation_c2w) > 2 else None
            lon, lat, alt = utm_to_lonlat(easting, northing, z, epsg_number=epsg_number)
            geometry_coords = [lon, lat, alt]
        except Exception as e:
            # 若转换失败，保留 None 并打印警告
            print(f"[WARN] UTM->lon/lat 转换失败（镜头 {shot_name}），错误: {e}")
            geometry_coords = None

        # ---- 5. 组织 Feature ----
        feature = {
            "type": "Feature",
            # id 可使用原名或数值索引，这里用索引字符串（你可以按需改）
            "id": str(idx),
            "name": shot_name,
            "imgId": str(start_img_id + idx),
            "properties": {
                "filename": shot_name,
                "focal_pixels": focal_out,
                "width": width,
                "height": height,
                "capture_time": capture_time_ms,
                # Cesium 要求 translation 为相机在世界坐标系的位置（与你示例一致）
                "translation": [float(translation_c2w[0]), float(translation_c2w[1]), float(translation_c2w[2])],
                # rotation 这里按 Cesium 的 [heading, pitch, roll] 顺序（度）
                "start_gps": start_gps,
                "end_gps": end_gps
            },
            "geometry": {
                "type": "Point",
                "coordinates": geometry_coords  # 可能为 None
            }
        }

        features.append(feature)
    # ---- 6. 写入 JSON 文件 ----
    data_dict = {
        "type": "FeatureCollection",
        "features": features
    }
    with outpt_path.open("w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2)


    return features