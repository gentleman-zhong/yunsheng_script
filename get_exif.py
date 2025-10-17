import logging
import re
import os
from PIL import Image
import xmltodict as x2d
from xml.parsers.expat import ExpatError
import json
import argparse

import exifread
from six import string_types
from datetime import datetime, timedelta
import pytz

# 仅保留parse_exif_values依赖的GPS参考模拟类
class GPSRefMock:
    def __init__(self, ref):
        self.values = [ref]

# --------- 精简后的 ODM_Photo（含 camera_make/model 与 YPR 修正） ----------
class ODM_Photo:
    def __init__(self, path_file):
        self.filename = os.path.basename(path_file)

        # 需要的字段
        self.camera_make = None
        self.camera_model = None

        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.yaw = None
        self.pitch = None
        self.roll = None

        logging.getLogger('exifread').setLevel(logging.CRITICAL)
        self.parse_exif_values(path_file)

    def parse_exif_values(self, _path_file):
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
                    self._set_attr_from_xmp_tag('yaw',  xtags, ['@drone-dji:FlightYawDegree', '@Camera:Yaw', 'Camera:Yaw'], float)
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

    # -------- 辅助函数（同前；保持与 parse 兼容） ----------
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


def generate_piclist(picPath: str, picNameList: list):
    """
    根据指定图片列表，生成包含经纬度、高程、姿态角信息的字典。
    
    参数:
        picPath (str): 图片所在目录路径
        picNameList (list): 需要提取的图片文件名列表 (如 ["1.jpg", "2.jpg"])
    
    返回:
        dict: {"pic0": [lat, lon, alt, roll, pitch, yaw], ...}
    """
    piclist = {}
    
    for idx, name in enumerate(picNameList):
        file_path = os.path.join(picPath, name)
        if not os.path.exists(file_path):
            print(f"⚠️ 文件未找到: {file_path}，跳过")
            continue

        try:
            photo = ODM_Photo(file_path)
            lat = getattr(photo, 'latitude', None)
            lon = getattr(photo, 'longitude', None)
            alt = getattr(photo, 'altitude', None)
            roll = getattr(photo, 'roll', None)
            pitch = getattr(photo, 'pitch', None)
            yaw = getattr(photo, 'yaw', None)
            
            # 将提取的数据写入结果字典
            piclist[name] = [lat, lon, alt, roll, pitch, yaw]

        except Exception as e:
            print(f"❌ 解析失败 {name}: {e}")

    return piclist


def get_exif_data(images_dir, images_list=[]):  
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.gif')
    picPath = os.path.join(images_dir,"images")
    picNameList = images_list
    if picNameList == []:
        picNameList = os.listdir(picPath)
        picNameList = [name for name in picNameList if os.path.splitext(name)[1].lower() in supported_extensions]


    result = generate_piclist(picPath, picNameList)

    # 输出 JSON 结果
    # result_json = json.dumps(result, indent=4, ensure_ascii=False)

    return result