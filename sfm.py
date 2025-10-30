import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


from pyproj.database import query_utm_crs_info
import numpy as np
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.transformer import Transformer
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from math import atan2, asin, degrees
import xml.etree.ElementTree as ET

import json

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


class SFM:
    def __init__(self, opensfm_path):
        self.opensfm_path = opensfm_path
        self.reconstruction_file = os.path.join(self.opensfm_path, "reconstruction.json")
        self.all_cameras = {}
        self.all_shots = {}
        self.all_points = {}
        self.all_reference_lla = []
        self.read_reconstruction_info()
        self.get_camera_model_pixel()

        self.ori_utm = self.convert_origin(self.all_reference_lla[0])
        self.espg_number = 32650
        self.offset = [0.0, 0.0, 0.0]
        self.get_espg_number(os.path.join(os.path.dirname(opensfm_path), "odm_georeferencing/coords.txt"))
    
    def read_reconstruction_info(self):
        with open(self.reconstruction_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data 可能是包含多个 reconstruction 的列表
        for rec_info in data:
            cameras_info = rec_info.get("cameras", {}) or {}
            shots_info = rec_info.get("shots", {}) or {}
            self.all_points = rec_info.get("points", {}) or {}
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

    def paint_features(self, image_dir, feature_dir, output_dir, max_workers=40):
        os.makedirs(output_dir, exist_ok=True)

        def process_single_image(feature_file):
            if not feature_file.endswith(".features.npz"):
                return

            feature_path = os.path.join(feature_dir, feature_file)
            data = np.load(feature_path, allow_pickle=True)

            points = data['points']  # shape (N,4): x, y, size, angle

            image_name = feature_file.replace(".features.npz", "")
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                print(f"图像不存在: {image_name}")
                return

            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图像: {image_name}")
                return

            h, w = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 绘制特征点（实心圆）
            for p in points:
                x_norm, y_norm, size, angle = p
                x_pix = int(round((w-1)/2 + x_norm * max(w, h)))
                y_pix = int(round((h-1)/2 + y_norm * max(w, h)))
                color = (255, 0, 0)
                cv2.circle(img, (x_pix, y_pix), radius=5, color=color, thickness=-1)

            output_path = os.path.join(output_dir, image_name)
            success = cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not success:
                print(f"[ERROR] 保存失败: {output_path}")
            else:
                print(f"[OK] 已保存: {output_path}")

        # 使用线程池并行处理每张图片
        feature_files = os.listdir(feature_dir)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_image, f) for f in feature_files]
            for _ in as_completed(futures):
                pass

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

    def get_reprojection_error(self, point3d, shot_data, camera_data, pixel):
        r_matrix = R.from_rotvec(shot_data["rotation"]).as_matrix()
        t_local = np.array(shot_data["translation"])
        M = np.eye(4)
        M[:3, :3] = r_matrix
        M[:3, 3] = t_local

        x_rpj, y_rpj = self.reproject_brown(camera_data, M, point3d)
        x , y = pixel
        return np.hypot(x - x_rpj, y - y_rpj)

    def reproject_brown(self, cam_data, M, coords_world):
        """Brown 模型投影"""
        pc = M @ np.array([coords_world[0], coords_world[1], coords_world[2], 1.0])

        xn, yn = pc[0] / pc[2], pc[1] / pc[2]
        r2 = xn**2 + yn**2
        dr = 1 + cam_data['k1']*r2 + cam_data['k2']*r2**2 + cam_data['k3']*r2**3
        dtx = 2*cam_data['p1']*xn*yn + cam_data['p2']*(r2 + 2*xn**2)
        dty = 2*cam_data['p2']*xn*yn + cam_data['p1']*(r2 + 2*yn**2)

        u = cam_data['focal_x'] * (dr*xn + dtx) + cam_data['c_x']
        v = cam_data['focal_y'] * (dr*yn + dty) + cam_data['c_y']
        return u, v

    def read_gcp_file(self):
        data = []
        with open(os.path.join(self.opensfm_path, "gcp_list.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 第一行是坐标系
        crs = lines[0].strip()

        # 解析后续行
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 7:
                continue  # 跳过异常行

            X, Y, Z = map(float, parts[:3])          # 世界坐标 (UTM)
            img_x, img_y = map(float, parts[3:5])    # 像素坐标
            image_name = parts[5]                    # 图像文件名
            gcp_id = parts[6]                        # 控制点ID

            data.append({
                "X": X,
                "Y": Y,
                "Z": Z,
                "img_x": img_x,
                "img_y": img_y,
                "image_name": image_name,
                "gcp_id": gcp_id
            })
        return crs, data

    def xml_to_gcp(self, xml_file, gcp_file):
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

    def write_metadata_xml(self, ori_utm, metadeata_xml_path):
        # 构建 XML 结构
        root = ET.Element("ModelMetadata", version="1")

        srs = ET.SubElement(root, "SRS")
        srs.text = "EPSG:32650"

        srs_origin = ET.SubElement(root, "SRSOrigin")
        srs_origin.text = f"{float(ori_utm[0])},{float(ori_utm[1])},{float(ori_utm[2])}"

        texture = ET.SubElement(root, "Texture")
        color_source = ET.SubElement(texture, "ColorSource")
        color_source.text = "Visible"

        # 写入 XML 文件
        tree = ET.ElementTree(root)
        tree.write(metadeata_xml_path, encoding='utf-8', xml_declaration=True)

# sfm = SFM()
# sfm.paint_features(image_dir="/home/zhangzhong/experiment/only_test/images",
#                   feature_dir="/home/zhangzhong/experiment/only_test/opensfm/features",
#                   output_dir="/home/zhangzhong/experiment/only_test/opensfm/features/output")

