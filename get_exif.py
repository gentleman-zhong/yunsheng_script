import logging
import re
import os
from PIL import Image
import xmltodict as x2d
from xml.parsers.expat import ExpatError
import json
import argparse

from pyproj.database import query_utm_crs_info
import numpy as np
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.transformer import Transformer
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from math import atan2, asin, degrees

import exifread
from six import string_types
from datetime import datetime, timedelta
import pytz

# 仅保留parse_exif_values依赖的GPS参考模拟类
class GPSRefMock:
    def __init__(self, ref):
        self.values = [ref]

class PhotoCorruptedException(Exception):
    pass

# --------- 精简后的 ODM_Photo（含 camera_make/model 与 YPR 修正） ----------
class ODM_Photo:
    def __init__(self, path_file):
        self.filename = os.path.basename(path_file)

        self.width = None
        self.height = None

        # 需要的字段
        self.camera_make = None
        self.camera_model = None

        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.rotation = None

        self.focal_ratio = 0.85

        logging.getLogger('exifread').setLevel(logging.CRITICAL)
        self.parse_exif_values(path_file)

    def parse_exif_values(self, _path_file):

        try:
            self.width, self.height = self.get_image_size(_path_file)
        except Exception as e:
            raise PhotoCorruptedException(str(e))

        tags = {}
        last_xtags = None

        with open(_path_file, 'rb') as f:
            tags = exifread.process_file(f, details=True, extract_thumbnail=False)

            # 读取相机厂商/型号（EXIF优先）
            try:
                if 'Image Make' in tags:
                    self.camera_make = str(tags['Image Make'].values).strip()
            except Exception:
                pass
            try:
                if 'Image Model' in tags:
                    self.camera_model = str(tags['Image Model'].values).strip()
            except Exception:
                pass

            # GPS altitude
            if 'GPS GPSAltitude' in tags:
                self.altitude = self._float_value(tags['GPS GPSAltitude'])
                if 'GPS GPSAltitudeRef' in tags and self._int_value(tags['GPS GPSAltitudeRef']) and self._int_value(tags['GPS GPSAltitudeRef']) > 0:
                    self.altitude = -self.altitude

            # GPS lat / lon (DMS -> decimal)
            if 'GPS GPSLatitude' in tags:
                ref = tags.get('GPS GPSLatitudeRef', GPSRefMock('N'))
                self.latitude = self._dms_to_decimal(tags['GPS GPSLatitude'], ref)
            if 'GPS GPSLongitude' in tags:
                ref = tags.get('GPS GPSLongitudeRef', GPSRefMock('E'))
                self.longitude = self._dms_to_decimal(tags['GPS GPSLongitude'], ref)

            # 解析 XMP（用于读取 yaw/pitch/roll，部分相机在 XMP 中保存姿态或厂商字段）
            f.seek(0)
            xmp_list = self._get_xmp(f)
            for xtags in xmp_list:
                last_xtags = xtags
                try:
                    # 先尝试读取 camera make/model（XMP 作为备选）
                    if not self.camera_make:
                        v = self._get_xmp_tag(xtags, ['@tiff:Make', 'tiff:Make', 'Make'])
                        if v:
                            self.camera_make = v
                    if not self.camera_model:
                        v = self._get_xmp_tag(xtags, ['@tiff:Model', 'tiff:Model', 'Model'])
                        if v:
                            self.camera_model = v

                    # 读取 Y/P/R（常见标签）
                    self._set_attr_from_xmp_tag('yaw',  xtags, ['@drone-dji:GimbalYawDegree', '@Camera:Yaw', 'Camera:Yaw'], float)
                    self._set_attr_from_xmp_tag('pitch',xtags, ['@drone-dji:GimbalPitchDegree', '@Camera:Pitch', 'Camera:Pitch'], float)
                    self._set_attr_from_xmp_tag('roll', xtags, ['@drone-dji:GimbalRollDegree', '@Camera:Roll', 'Camera:Roll'], float)

                    # 如果三者都存在，做厂商特定修正与标准化（遵循你提供的原始逻辑）
                    if self._has_ypr():
                        make = (self.camera_make or "").lower()
                        # DJI / Hasselblad 特殊修正（原始代码里是先 +90 再统一 -90）
                        if make in ['dji', 'hasselblad', 'ikingtec']:
                            # 保持与原始逻辑一致：先 +90，再整体 -90（效果上等同于原值，但保留以避免行为差异）
                            self.pitch = 90.0 + float(self.pitch)
                        # SenseFly 需要翻转 roll
                        if make == 'sensefly':
                            try:
                                self.roll = -float(self.roll)
                            except Exception:
                                pass
                        # 俯仰角标准化：统一减去90（原始代码有这一步）
                        try:
                            self.pitch = float(self.pitch) - 90.0
                        except Exception:
                            pass

                        # 一旦完成修正，结束 xmp 循环
                        break
                except Exception:
                    # 容错：继续尝试下一个 xmp 块
                    continue
        
        if tags is not None and last_xtags is not None:
            self.compute_focal(tags, last_xtags) 

        self.rotation = self.EulerAnglesToRotationVector([self.roll, self.pitch, self.yaw])

    def EulerAnglesToRotationVector(self, _angles):
        # ENU to NED
        A1 = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1]])
        # NED to cam
        A2 = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])

        r = - np.pi * _angles[0] / 180
        p = - np.pi * _angles[1] / 180
        y = - np.pi * _angles[2] / 180
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])
        Rot = np.dot(Rx, np.dot(Ry, Rz))

        M = np.dot(A2, np.dot(Rot, A1))
        V = R.from_matrix(M).as_rotvec()
        return V

    def _set_attr_from_xmp_tag(self, attr, xmp_tags, tags, cast=None):
        v = self._get_xmp_tag(xmp_tags, tags)
        if v is not None:
            try:
                if cast is not None:
                    if isinstance(v, str) and "/" in v and cast in (int, float):
                        v = self._try_parse_fraction(v)
                    setattr(self, attr, cast(v))
                else:
                    setattr(self, attr, v)
            except (ValueError, TypeError):
                pass

    def _get_xmp_tag(self, xmp_tags, tags):
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag in xmp_tags:
                t = xmp_tags[tag]
                if isinstance(t, str):
                    return t.strip()
                if isinstance(t, (int, float)):
                    return t
                if isinstance(t, dict):
                    items = t.get('rdf:Seq', {}).get('rdf:li', {})
                    if isinstance(items, str):
                        return items.strip()
                    if isinstance(items, list):
                        return " ".join(str(x) for x in items)
        return None

    def _get_xmp(self, fileobj):
        data = fileobj.read()
        start = data.find(b'<x:xmpmeta')
        end = data.find(b'</x:xmpmeta')
        if 0 <= start < end:
            s = data[start:end+12].decode('utf8', errors='ignore')
            try:
                xdict = x2d.parse(s)
            except ExpatError:
                from bs4 import BeautifulSoup
                s2 = str(BeautifulSoup(s, 'xml'))
                xdict = x2d.parse(s2)
            xdict = xdict.get('x:xmpmeta', {}).get('rdf:RDF', {}).get('rdf:Description', {})
            return xdict if isinstance(xdict, list) else [xdict]
        return []

    def _dms_to_decimal(self, dms_tag, sign):
        deg, minute, sec = self._float_values(dms_tag)
        if deg is None or minute is None or sec is None:
            return None
        decimal = deg + minute/60.0 + sec/3600.0
        if getattr(sign, 'values', [None])[0] in ('S','s','W','w'):
            return -decimal
        return decimal

    def _float_values(self, tag):
        if not hasattr(tag, 'values'):
            return (None, None, None)
        res = []
        for v in tag.values:
            if isinstance(v, (int, float)):
                res.append(float(v))
            elif hasattr(v, 'num') and hasattr(v, 'den') and v.den != 0:
                res.append(float(v.num) / float(v.den))
            else:
                try:
                    res.append(float(str(v)))
                except Exception:
                    res.append(None)
        while len(res) < 3:
            res.append(None)
        return res[0], res[1], res[2]

    def _float_value(self, tag):
        vals = self._float_values(tag)
        return vals[0] if vals and vals[0] is not None else None

    def _int_value(self, tag):
        if not hasattr(tag, 'values'):
            return None
        try:
            for v in tag.values:
                if isinstance(v, int):
                    return int(v)
                if isinstance(v, str) and v.isdigit():
                    return int(v)
            return None
        except Exception:
            return None

    def _try_parse_fraction(self, val):
        try:
            a,b = val.split('/')
            return float(a)/float(b) if float(b) != 0 else float(val)
        except Exception:
            return float(val)

    def _has_ypr(self):
        return (self.yaw is not None) and (self.pitch is not None) and (self.roll is not None)

    def get_image_size(self, file_path, fallback_on_error=True):
        """
        Return (width, height) for a given img file
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as e:
            if fallback_on_error:
                print(f"⚠️ 读取图片尺寸失败: {file_path}，使用 OpenCV 代替")
                import cv2
                img = cv2.imread(file_path)
                width = img.shape[1]
                height = img.shape[0]
            else:
                raise e

        return (width, height)
    
    def compute_focal(self, tags, xtags):
        try:
            self.focal_ratio = self.extract_focal(self.camera_make, self.camera_model, tags, xtags)
        except (IndexError, ValueError) as e:
            print(f"⚠️ 计算焦距失败: {e}")

    def extract_focal(self, make, model, tags, xtags):
        if make != "unknown":
            # remove duplicate 'make' information in 'model'
            model = model.replace(make, "")
        

        sensor_width = None
        if ("EXIF FocalPlaneResolutionUnit" in tags and "EXIF FocalPlaneXResolution" in tags):
            resolution_unit = self._float_value(tags["EXIF FocalPlaneResolutionUnit"])
            mm_per_unit = self.get_mm_per_unit(resolution_unit)
            if mm_per_unit:
                pixels_per_unit = self._float_value(tags["EXIF FocalPlaneXResolution"])
                if pixels_per_unit <= 0 and "EXIF FocalPlaneYResolution" in tags:
                    pixels_per_unit = self._float_value(tags["EXIF FocalPlaneYResolution"])
                
                if pixels_per_unit > 0 and self.width is not None:
                    units_per_pixel = 1 / pixels_per_unit
                    sensor_width = self.width * units_per_pixel * mm_per_unit

        focal_35 = None
        focal = None
        if "EXIF FocalLengthIn35mmFilm" in tags:
            focal_35 = self._float_value(tags["EXIF FocalLengthIn35mmFilm"])
        if "EXIF FocalLength" in tags:
            focal = self._float_value(tags["EXIF FocalLength"])
        if focal is None and "@aux:Lens" in xtags:
            lens = self._get_xmp_tag(xtags, ["@aux:Lens"])
            matches = re.search('([\d\.]+)mm', str(lens))
            if matches:
                focal = float(matches.group(1))

        if focal_35 is not None and focal_35 > 0:
            focal_ratio = focal_35 / 36.0  # 35mm film produces 36x24mm pictures.
        else:
            if sensor_width and focal:
                focal_ratio = focal / sensor_width
            else:
                focal_ratio = 0.85

        return focal_ratio
    
    def get_mm_per_unit(self, resolution_unit):
        """Length of a resolution unit in millimeters.

        Uses the values from the EXIF specs in
        https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html

        Args:
            resolution_unit: the resolution unit value given in the EXIF
        """
        if resolution_unit == 2:  # inch
            return 25.4
        elif resolution_unit == 3:  # cm
            return 10
        elif resolution_unit == 4:  # mm
            return 1
        elif resolution_unit == 5:  # um
            return 0.001
        else:
            print("Unknown EXIF resolution unit value: {}".format(resolution_unit))
            return None

def get_camera_model_from_exif(width, height, focal_ratio, projection='brown',
                               k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
    """根据相机参数生成相机模型字典"""
    try:
        W = float(width)
        H = float(height)
        fr = float(focal_ratio) if focal_ratio is not None else 0.0
    except Exception:
        raise ValueError("width, height and focal_ratio must be numeric")

    # 像素焦距
    focal_px = fr * W
    camera_dict = {
        "projection_type": projection,
        "width": int(W),
        "height": int(H),
        "focal_x": float(focal_px),
        "focal_y": float(focal_px),
        "c_x": float((W - 1) / 2.0),
        "c_y": float((H - 1) / 2.0),
        "k1": float(k1),
        "k2": float(k2),
        "p1": float(p1),
        "p2": float(p2),
        "k3": float(k3)
    }
    return camera_dict

def generate_piclist(picPath: str, picNameList: list):
    piclist = {}
    all_cameras = {}
    for idx, name in enumerate(picNameList):
        file_path = os.path.join(picPath, name)
        if not os.path.exists(file_path):
            print(f"⚠️ 文件未找到: {file_path}，跳过")
            continue

        try:
            photo = ODM_Photo(file_path)

            piclist[name] = {}

            lat = getattr(photo, 'latitude', None)
            lon = getattr(photo, 'longitude', None)
            alt = getattr(photo, 'altitude', None)
            rotation = getattr(photo, 'rotation', None)
            # 将提取的数据写入结果字典
            piclist[name]["pos"] = [lat, lon, alt, rotation[0], rotation[1], rotation[2]]

            # 记录相机型号
            width = getattr(photo, 'width', None)
            height = getattr(photo, 'height', None)
            focal_ratio = getattr(photo, 'focal_ratio', None)
            camera_make = getattr(photo, 'camera_make', None)
            camera_model = getattr(photo, 'camera_model', None)

            if not width or not height or not focal_ratio:
                print(f"⚠️ 相机信息不完整，跳过 {name}")
                continue

            make_clean = (camera_make or "unknown").strip().lower().replace(" ", "_")
            model_clean = (camera_model or "unknown").strip().lower().replace(" ", "_")
            key = f"{make_clean}_{model_clean}_{int(width)}x{int(height)}_brown_{focal_ratio:.5f}"

            piclist[name]["cam_id"] = key
            
            if key not in all_cameras:
                all_cameras[key] = get_camera_model_from_exif(width, height, focal_ratio)

        except Exception as e:
            print(f"❌ 解析失败 {name}: {e}")

    return piclist, all_cameras

class ODM_Result:
    def __init__(self, opensfm_path):
        self.reconstruction_file = os.path.join(opensfm_path, "reconstruction.json")
        self.all_cameras = {}
        self.all_shots = {}
        self.all_reference_lla = []
        self.read_reconstruction_info(self.reconstruction_file)
        self.get_camera_model_pixel()

        self.ori_utm = self.convert_origin(self.all_reference_lla[0])
        self.espg_number = 32650
        self.offset = [0.0, 0.0, 0.0]
        self.get_espg_number(os.path.join(os.path.dirname(opensfm_path), "odm_georeferencing/coords.txt"))
    
    def read_reconstruction_info(self, reconstruction_file: str):
        with open(reconstruction_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data 可能是包含多个 reconstruction 的列表
        for rec_info in data:
            cameras_info = rec_info.get("cameras", {}) or {}
            shots_info = rec_info.get("shots", {}) or {}
            reference_lla = rec_info.get("reference_lla", None)

            # 记录参考坐标
            if reference_lla is not None:
                self.all_reference_lla.append(reference_lla)

            # 合并 cameras（遇到重复 name 时保留首次出现的）
            for camera_name, camera_data in cameras_info.items():
                if camera_name not in self.all_cameras:
                    self.all_cameras[camera_name] = camera_data

            # 合并 shots（遇到重复 name 时保留首次出现的）
            for shot_name, shot_data in shots_info.items():
                if shot_name not in self.all_shots:
                    self.all_shots[shot_name] = shot_data

    def convert_origin(self, _origin):
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

    def get_one_shot_info(self, rotation, translation, ori_utm):
        r_matrix = R.from_rotvec(rotation).as_matrix()
        
        ori_utm = np.array([ori_utm[0], ori_utm[1], 0.0])
        # 更新平移向量
        t_new = translation - r_matrix @ ori_utm
        
        # 构建 4x4 位姿矩阵
        M = np.eye(4)
        M[:3, :3] = r_matrix
        M[:3, 3] = t_new
        M_inv = np.linalg.inv(M)

        # 世界坐标系下，相机原点的位置，utm形式
        t_c2w = M_inv[:3, 3]
        camera_loc_gps = self.utm_to_lonlat(t_c2w[0], t_c2w[1], t_c2w[2], epsg_number=self.espg_number)
        
        return camera_loc_gps, rotation 

    def utm_to_lonlat(self, easting, northing, height=None, epsg_number=32650):
        src_crs = f"EPSG:{epsg_number}"
        dst_crs = "EPSG:4326"  # WGS84 lon/lat
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        if height is None:
            return float(lat), float(lon), None
        else:
            return float(lat), float(lon), float(height)

    def get_espg_number(self, coords_path):
        if os.path.exists(coords_path):
            espg_offset_dirct = self.parse_coords_file(coords_path)
            self.espg_number = espg_offset_dirct["epsg"]
            self.offset = espg_offset_dirct["offset"]
    
    def parse_coords_file(self, coords_file: str):
        coords_file = Path(coords_file)
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

    def get_camera_model_pixel(self):
        for camera_name, camera_data in self.all_cameras.items():
            width = camera_data['width']
            height = camera_data['height']
            focal_x = camera_data['focal_x'] * max(width, height)
            focal_y = camera_data['focal_y'] * max(width, height)
            c_x = camera_data['c_x'] * max(width, height) + (width - 1) / 2
            c_y = camera_data['c_y'] * max(width, height) + (height - 1) / 2
            k1 = camera_data['k1']
            k2 = camera_data['k2']
            p1 = camera_data['p1']
            p2 = camera_data['p2']
            k3 = camera_data['k3']
            camera_dict = {
                "projection_type": camera_data['projection_type'],
                "width": int(width),
                "height": int(height),
                "focal_x": float(focal_x),
                "focal_y": float(focal_y),
                "c_x": float(c_x),
                "c_y": float(c_y),
                "k1": float(k1),
                "k2": float(k2),
                "p1": float(p1),
                "p2": float(p2),
                "k3": float(k3)
            }
            self.all_cameras[camera_name] = camera_dict

    def get_all_output_shots(self, picNameList):
        output_all_shots = {}
        for shot_name, shot_data in self.all_shots.items():
            if shot_name not in picNameList:
                continue
            output_all_shots[shot_name] = {}
            rotation = shot_data['rotation']
            translation = shot_data['translation']
            camera_name = shot_data['camera']
            gps_pos,  rotation = self.get_one_shot_info(rotation, translation, self.ori_utm)
            output_all_shots[shot_name]["pos"] = list([gps_pos[0], gps_pos[1], gps_pos[2], rotation[0], rotation[1], rotation[2]])
            output_all_shots[shot_name]["cam_id"] = camera_name
        return output_all_shots

def get_exif_data(images_dir, images_list=[]):  
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.gif')
    picPath = os.path.join(images_dir,"images")
    picNameList = images_list
    if picNameList == []:
        picNameList = os.listdir(picPath)
        picNameList = [name for name in picNameList if os.path.splitext(name)[1].lower() in supported_extensions]


    piclist, all_cameras = generate_piclist(picPath, picNameList)

    return piclist, all_cameras

def get_reconstruction_data(images_dir, images_list=[]):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.gif')
    picPath = os.path.join(images_dir,"images")
    picNameList = images_list
    if picNameList == []:
        picNameList = os.listdir(picPath)
        picNameList = [name for name in picNameList if os.path.splitext(name)[1].lower() in supported_extensions]
    opensfm_dir = os.path.join(images_dir,"opensfm")
    odm_result =   ODM_Result(opensfm_dir)
    return odm_result.get_all_output_shots(picNameList), odm_result.all_cameras

    
def get_gcp_info(images_dir, images_list=[]):
    rec_file = os.path.join(images_dir,"opensfm", "reconstruction.json")
    if not os.path.exists(rec_file):
        piclist, all_cameras = get_exif_data(images_dir, images_list)
    else:
        piclist, all_cameras = get_reconstruction_data(images_dir, images_list)
    return piclist, all_cameras