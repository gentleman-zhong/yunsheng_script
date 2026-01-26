import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from pyproj import CRS
from pyproj import Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
from scipy.spatial.transform import Rotation as R
import json
import sys
import argparse
import debugpy
import os
import csv
import struct


# p1,p2需要取反
# 主点位置减一
# focal_pixel减一
def init_root(_ori):
  root = ET.Element('BlocksExchange')
  root.set('version', '3.2')

  spatial_reference_systems = ET.SubElement(root, 'SpatialReferenceSystems')

  srs_wgs84 = ET.SubElement(spatial_reference_systems, 'SRS')
  srs_wgs84_id = ET.SubElement(srs_wgs84, 'Id')
  srs_wgs84_id.text = '1'
  srs_wgs84_name = ET.SubElement(srs_wgs84, 'Name')
  srs_wgs84_name.text = 'WGS 84 (EPSG:4326)'
  srs_wgs84_definition = ET.SubElement(srs_wgs84, 'Definition')
  srs_wgs84_definition.text = 'WGS84'
  
  srs_utm = ET.SubElement(spatial_reference_systems, 'SRS')
  srs_utm_id = ET.SubElement(srs_utm, 'Id')
  srs_utm_id.text = '2'
  srs_utm_name = ET.SubElement(srs_utm, 'Name')
  srs_utm_name.text = 'WGS 84 / UTM zone 50N (EPSG:32650)'
  srs_utm_definition = ET.SubElement(srs_utm, 'Definition')
  srs_utm_definition.text = 'EPSG:32650'

  block = ET.SubElement(root, 'Block')

  block_name = ET.SubElement(block, 'Name')
  block_name.text = 'ODM - AT'
  block_description = ET.SubElement(block, 'Description')
  block_description.text = 'ODM converts to CC'
  # if not np.array_equal(_ori, np.array([0., 0., 0.])):
  #   block_srsid = ET.SubElement(block, 'SRSId')
  #   block_srsid.text = '2'
  photogroups = ET.SubElement(block, 'Photogroups')
  tiepoints = ET.SubElement(block, 'TiePoints')
  return root

def add_photogroup(_root, _data,  _name):
  photogroups = _root.find('Block').find('Photogroups')
  photogroup = ET.SubElement(photogroups, 'Photogroup')
  
  photogroup_name = ET.SubElement(photogroup, 'Name')
  photogroup_name.text = _name
  photogroup_manualopticalparams = ET.SubElement(photogroup, 'ManualOpticalParams')
  photogroup_manualopticalparams.text = 'true'
  photogroup_manualpose = ET.SubElement(photogroup, 'ManualPose')
  photogroup_manualpose.text = 'true'
  photogroup_imagedimensions = ET.SubElement(photogroup, 'ImageDimensions')

  imagedimensions_width = ET.SubElement(photogroup_imagedimensions, 'Width')
  imagedimensions_width.text = str(_data['width'])
  imagedimensions_height = ET.SubElement(photogroup_imagedimensions, 'Height')
  imagedimensions_height.text = str(_data['height'])

  normalizer = max(_data['width'], _data['height'])
  # normalizer = _data['width']
  photogroup_cameramodeltype = ET.SubElement(photogroup, 'CameraModelType')
  photogroup_cameramodeltype.text = 'Perspective'
  photogroup_cameramodelband = ET.SubElement(photogroup, 'CameraModelBand')
  photogroup_cameramodelband.text = 'Visible'
  photogroup_focallength= ET.SubElement(photogroup, 'FocalLengthPixels')
  photogroup_focallength.text = str(_data['focal'] * normalizer) if 'focal' in _data else str(_data['focal_x'] * normalizer)
  photogroup_cameraorientation = ET.SubElement(photogroup, 'CameraOrientation')
  photogroup_cameraorientation.text = 'XRightYDown'

  photogroup_principalpoint = ET.SubElement(photogroup, 'PrincipalPoint')

  principalpoint_x = ET.SubElement(photogroup_principalpoint, 'x')
  principalpoint_x.text = str((_data['width'] -1) / 2 + _data['c_x'] * normalizer)
  principalpoint_y = ET.SubElement(photogroup_principalpoint, 'y')
  principalpoint_y.text = str((_data['height'] -1)/ 2 + _data['c_y'] * normalizer) 

  photogroup_distortion = ET.SubElement(photogroup, 'Distortion')

  distortion_k1 = ET.SubElement(photogroup_distortion, 'K1')
  distortion_k1.text = str(_data['k1'])
  distortion_k2 = ET.SubElement(photogroup_distortion, 'K2')
  distortion_k2.text = str(_data['k2'])
  distortion_k3 = ET.SubElement(photogroup_distortion, 'K3')
  distortion_k3.text = str(_data['k3'])
  distortion_p1 = ET.SubElement(photogroup_distortion, 'P1')
  distortion_p1.text = str(_data['p2'])
  distortion_p2 = ET.SubElement(photogroup_distortion, 'P2')
  distortion_p2.text = str(_data['p1'])

  photogroup_aspectratio = ET.SubElement(photogroup, 'AspectRatio')
  photogroup_aspectratio.text = '1'
  photogroup_skew = ET.SubElement(photogroup, 'Skew')
  photogroup_skew.text = '0'

def add_photo(_root, _data, _photogroup_name):
  photogroups = _root.find('Block').find('Photogroups')
  for photogroup in photogroups.findall('Photogroup'):
    photogroup_name = photogroup.find('Name')
    if photogroup_name is not None and photogroup_name.text == _photogroup_name:
      break

  photo = ET.SubElement(photogroup, 'Photo')

  photo_id = ET.SubElement(photo, 'Id')
  photo_id.text = str(_data['Id'])
  photo_imagepath = ET.SubElement(photo, 'ImagePath')
  
  photo_imagepath.text = (_data['Name'])

  photo_component = ET.SubElement(photo, 'Component')
  photo_component.text = '1'

  pose = ET.SubElement(photo, 'Pose')

  rotation = ET.SubElement(pose, 'Rotation')

  rotation_m00 = ET.SubElement(rotation, 'M_00')
  rotation_m00.text = str(_data['Rotation']['M_00'])
  rotation_m01 = ET.SubElement(rotation, 'M_01')
  rotation_m01.text = str(_data['Rotation']['M_01'])
  rotation_m02 = ET.SubElement(rotation, 'M_02')
  rotation_m02.text = str(_data['Rotation']['M_02'])
  rotation_m10 = ET.SubElement(rotation, 'M_10')
  rotation_m10.text = str(_data['Rotation']['M_10'])
  rotation_m11 = ET.SubElement(rotation, 'M_11')
  rotation_m11.text = str(_data['Rotation']['M_11'])
  rotation_m12 = ET.SubElement(rotation, 'M_12')
  rotation_m12.text = str(_data['Rotation']['M_12'])
  rotation_m20 = ET.SubElement(rotation, 'M_20')
  rotation_m20.text = str(_data['Rotation']['M_20'])
  rotation_m21 = ET.SubElement(rotation, 'M_21')
  rotation_m21.text = str(_data['Rotation']['M_21'])
  rotation_m22 = ET.SubElement(rotation, 'M_22')
  rotation_m22.text = str(_data['Rotation']['M_22'])
  rotation_accurate = ET.SubElement(rotation, 'Accurate')
  rotation_accurate.text = 'true'

  center = ET.SubElement(pose, 'Center')

  center_x = ET.SubElement(center, 'x')
  center_x.text = str(_data['Center']['x'])
  center_y = ET.SubElement(center, 'y')
  center_y.text = str(_data['Center']['y'])
  center_z = ET.SubElement(center, 'z')
  center_z.text = str(_data['Center']['z'])
  center_accurate = ET.SubElement(center, 'Accurate')
  center_accurate.text = 'true'

def add_tiepoint(_root, _data):
  tiepoints = _root.find('Block').find('TiePoints')

  tiepoint = ET.SubElement(tiepoints, 'TiePoint')

  position = ET.SubElement(tiepoint, 'Position')

  position_x = ET.SubElement(position, 'x')
  position_x.text = str(_data['Position']['x'])
  position_y = ET.SubElement(position, 'y')
  position_y.text = str(_data['Position']['y'])
  position_z = ET.SubElement(position, 'z')
  position_z.text = str(_data['Position']['z'])

  color = ET.SubElement(tiepoint, 'Color')

  color_x = ET.SubElement(color, 'Red')
  color_x.text = str(_data['Color']['Red'])
  color_y = ET.SubElement(color, 'Green')
  color_y.text = str(_data['Color']['Green'])
  color_z = ET.SubElement(color, 'Blue')
  color_z.text = str(_data['Color']['Blue'])

# 用字典去重：同一个 PhotoId 只保留第一次出现的
  unique_measurements = {}
  for m in _data['Measurement']:
      pid = m['PhotoId']
      if pid not in unique_measurements:
          unique_measurements[pid] = m

  # 写入 XML
  for m in unique_measurements.values():
      measurement = ET.SubElement(tiepoint, 'Measurement')
      measurement_photoid = ET.SubElement(measurement, 'PhotoId')
      measurement_photoid.text = str(m['PhotoId'])
      measurement_x = ET.SubElement(measurement, 'x')
      measurement_x.text = str(m['x'])
      measurement_y = ET.SubElement(measurement, 'y')
      measurement_y.text = str(m['y'])

# ----------------------------------------------------------------------------------------------------------------------------------------- #
#   .---.       ,-----.    ,---.  ,---.   .-''-.             .-```-.             .-------.     .-''-.     ____        _______      .-''-.   #
#   | ,_|     .'  .-,  '.  |   /  |   | .'_ _   \           /   _   \            \  _(`)_ \  .'_ _   \  .'  __ `.    /   __  \   .'_ _   \  #
# ,-./  )    / ,-.|  \ _ \ |  |   |  .'/ ( ` )   '         |  .` '\__|           | (_ o._)| / ( ` )   '/   '  \  \  | ,_/  \__) / ( ` )   ' #
# \  '_ '`) ;  \  '_ /  | :|  | _ |  |. (_ o _)  |          \  `--.              |  (_,_) /. (_ o _)  ||___|  /  |,-./  )      . (_ o _)  | #
#  > (_)  ) |  _`,/ \ _/  ||  _( )_  ||  (_,_)___|         /' ..--`.----.        |   '-.-' |  (_,_)___|   _.-`   |\  '_ '`)    |  (_,_)___| #
# (  .  .-' : (  '\_/ \   ;\ (_ o._) /'  \   .---.        :  `     ' ___|        |   |     '  \   .---..'   _    | > (_)  )  __'  \   .---. #
#  `-'`-'|___\ `"/  \  ) /  \ (_,_) /  \  `-'    /        |   `..-'_( )_         |   |      \  `-'    /|  _( )_  |(  .  .-'_/  )\  `-'    / #
#   |        \'. \_/``".'    \     /    \       /          \      (_ o _)        /   )       \       / \ (_ o _) / `-'`-'     /  \       /  #
#   `--------`  '-----'       `---`      `'-..-'            `-.__..(_,_)         `---'        `'-..-'   '.(_,_).'    `._____.'    `'-..-'   #
# ----------------------------------------------------------------------------------------------------------------------------------------- #

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

def convert_photo_data(_shots, _ori_utm): 
  photos = {}
  for i, (pic_name, pic_data) in enumerate(_shots.items()):
    pic_dic = {}
    pic_dic['Id'] = i
    pic_dic['Name'] = pic_name
    pic_dic['Camera'] = pic_data['camera']
    r_vector = pic_data['rotation']
    r_matrix = R.from_rotvec(r_vector).as_matrix()
    pic_dic['Rotation'] = {
      'M_00': r_matrix[0][0],
      'M_01': r_matrix[0][1],
      'M_02': r_matrix[0][2],
      'M_10': r_matrix[1][0],
      'M_11': r_matrix[1][1],
      'M_12': r_matrix[1][2],
      'M_20': r_matrix[2][0],
      'M_21': r_matrix[2][1],
      'M_22': r_matrix[2][2]
    }
    t_local = np.array(pic_data['translation'])
    M = np.hstack([np.vstack([r_matrix, np.zeros((3,))]), np.append(t_local, 1).reshape(4, 1)]) # 从世界坐标系（UTM）到相机坐标系变换矩阵
    M_inv = np.linalg.inv(M)
    t_world = np.dot(M_inv, np.array([0, 0, 0, 1]))[:3]
    p_world = _ori_utm + t_world
    pic_dic['Center'] = {
      'x': p_world[0],
      'y': p_world[1],
      'z': p_world[2]
    }
    photos[pic_name] = pic_dic
  
  return photos

def convert_tiepoints_data(_points, _tracks, _photos, _cams, _ori_utm):  
  tiepoints = {}
  # low_points = 0
  for pt_id, pt in _points.items():
    pt_dic = {}
    coods_local = np.array(pt['coordinates'])
    # if coods_local[2] < 0:
    #   continue
    coods_world = _ori_utm + coods_local
    pt_dic['Position'] = {
      'x': coods_world[0],
      'y': coods_world[1],
      'z': coods_world[2]
    }
    color_int = np.array(pt['color'])
    color_unit = color_int / 255
    pt_dic['Color'] = {
      'Red': color_unit[0],
      'Green': color_unit[1],
      'Blue': color_unit[2]
    }
    pic_list = _tracks[pt_id]

    measurements = []
    for pic in pic_list:
      pic_name = pic[0]
      u = float(pic[1])
      v = float(pic[2])
      if pic_name not in _photos:
        continue
      else:
        cam_name = _photos[pic_name]['Camera']
      cam_data = _cams[cam_name]
      w = cam_data['width']
      h = cam_data['height']
      normalizer = max(w, h)
      x = (w - 1) / 2 + u * normalizer
      y = (h - 1) / 2 + v * normalizer # both u and v are normalized to width
      measurement_dic = {}
      measurement_dic['PhotoId'] = _photos[pic_name]['Id']
      measurement_dic['x'] = x
      measurement_dic['y'] = y
      measurements.append(measurement_dic)
    pt_dic['Measurement'] = measurements
    tiepoints[pt_id] = pt_dic

  return tiepoints

def get_projection_matrix(_photo):
  r_dic = _photo['Rotation']
  r_m = np.zeros(9)
  r_m = r_m.reshape((3,3))
  r_m[0][0] = r_dic['M_00']
  r_m[0][1] = r_dic['M_01']
  r_m[0][2] = r_dic['M_02']
  r_m[1][0] = r_dic['M_10']
  r_m[1][1] = r_dic['M_11']
  r_m[1][2] = r_dic['M_12']
  r_m[2][0] = r_dic['M_20']
  r_m[2][1] = r_dic['M_21']
  r_m[2][2] = r_dic['M_22']
  t_dic = _photo['Center']
  t = np.zeros(3)
  t[0] = t_dic['x']
  t[1] = t_dic['y']
  t[2] = t_dic['z']
  M = np.hstack([np.vstack([r_m, np.zeros((3,))]), np.append(-1 * np.dot(r_m, t), 1).reshape(4, 1)])
  return M

def reproject_perspective(_cam_data, _photo, _coods_world):
  f = _cam_data['focal']
  k1 = _cam_data['k1']
  k2 = _cam_data['k2']

  M = get_projection_matrix(_photo)
  pc = np.dot(M, _coods_world)

  xn = pc[0] / pc[2]
  yn = pc[1] / pc[2]
  r2 = xn ** 2 + yn ** 2
  d = 1 + k1 * r2 + k2 * r2 * r2
  u_rpj = f * d * xn
  v_rpj = f * d * yn

  return u_rpj, v_rpj

def reproject_brown(_cam_data, _photo, _coods_world):
  fx = _cam_data['focal_x']
  fy = _cam_data['focal_y']
  cx = _cam_data['c_x']
  cy = _cam_data['c_y']
  k1 = _cam_data['k1']
  k2 = _cam_data['k2']
  p1 = _cam_data['p1']
  p2 = _cam_data['p2']
  k3 = _cam_data['k3']

  M = get_projection_matrix(_photo)
  pc = np.dot(M, _coods_world)

  xn = pc[0] / pc[2]
  yn = pc[1] / pc[2]
  r2 = xn ** 2 + yn ** 2
  dr = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
  dtx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
  dty = 2 * p2 * xn * yn + p1 * (r2 + 2 * yn * yn)
  u_rpj = fx * (dr * xn + dtx) + cx
  v_rpj = fy * (dr * yn + dty) + cy


  return u_rpj, v_rpj

def convert_and_filter_tiepoints_data(_points, _tracks, _photos, _cams, _ori_utm, _threshold, _trk_len):
  tiepoints = {}
  rpj_error_list = {} # debug
  for pt_id, pt in _points.items():
    pt_dic = {}
    coods_local = np.array(pt['coordinates'])
    coods_world = _ori_utm + coods_local
    pt_dic['Position'] = {
      'x': coods_world[0],
      'y': coods_world[1],
      'z': coods_world[2]
    }
    coods_world = np.append(coods_world, 1.)
    color_int = np.array(pt['color'])
    color_unit = color_int / 255
    pt_dic['Color'] = {
      'Red': color_unit[0],
      'Green': color_unit[1],
      'Blue': color_unit[2]
    }
    pic_list = _tracks[pt_id]

    measurements = []
    rpj_error_pt = {} # debug
    for pic in pic_list:
      pic_name = pic[0]
      u = float(pic[1])
      v = float(pic[2])
      if pic_name not in _photos:
        continue
      else:
        cam_name = _photos[pic_name]['Camera']
      cam_data = _cams[cam_name]
      prj_type = cam_data['projection_type']
      w = cam_data['width']
      h = cam_data['height']
      x = (w - 1) / 2 + u * max(w, h) # both u and v are normalized to max(w, h)
      y = (h - 1) / 2 + v * max(w, h)
      measurement_dic = {}
      measurement_dic['PhotoId'] = _photos[pic_name]['Id']
      measurement_dic['x'] = x
      measurement_dic['y'] = y 

      if prj_type == 'perspective':
        u_rpj, v_rpj = reproject_perspective(cam_data, _photos[pic_name], coods_world)
      elif prj_type == 'simple_radial':
        pass
      elif prj_type == 'radial':
        pass
      elif prj_type == 'brown':
        u_rpj, v_rpj = reproject_brown(cam_data, _photos[pic_name], coods_world)
      else:
        print('Error: projection_type unrecogonized!', prj_type)
        exit()
      x_rpj = (w - 1) / 2 + u_rpj * max(w, h) # both u and v are normalized to max(w, h)
      y_rpj = (h - 1) / 2 + v_rpj * max(w, h)

      # print(x, y, '----', x_rpj, y_rpj)
      rpjerror = np.sqrt((x - x_rpj) ** 2 + (y - y_rpj) ** 2)
      rpj_error_pt[pic_name] = rpjerror # debug
      if rpjerror > _threshold:
        continue

      measurements.append(measurement_dic)

    sorted_rpj_error_pt = dict(sorted(rpj_error_pt.items())) # debug
    rpj_error_list[pt_id] = sorted_rpj_error_pt # debug
    if len(measurements) < _trk_len:
      continue

    pt_dic['Measurement'] = measurements
    tiepoints[pt_id] = pt_dic
  return tiepoints

def read_tracks_binary(filename):
    with open(filename, 'rb') as f:
        # 1. 读取第一行（版本信息）
        version_line = f.readline().decode().strip()
        print(f"Tracks file version: {version_line}")
        
        records = []

        while True:
            # 2. 读取 TrackLengths（2个uint16_t = 4字节）
            lengths_bytes = f.read(4)
            if len(lengths_bytes) < 4:
                break  # 文件读完

            image_len, track_id_len = struct.unpack('<HH', lengths_bytes)  # 小端

            # 3. 读取 image 和 track ID
            str_len = image_len + track_id_len
            str_bytes = f.read(str_len)
            if len(str_bytes) < str_len:
                break

            image = str_bytes[:image_len].decode()
            track_id = str_bytes[image_len:].decode()

            # 4. 读取 TrackRecord（4+12+3=19字节）
            record_bytes = f.read(19)
            if len(record_bytes) < 19:
                break

            feature_id, x, y, scale, r, g, b = struct.unpack('<ifffBBB', record_bytes)

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

def get_metadata_xml(ori_utm, metadeata_xml_path):
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
# ----------------------------------------------------------------------------------------------------------------------------------------- #
#   .---.       ,-----.    ,---.  ,---.   .-''-.             .-```-.             .-------.     .-''-.     ____        _______      .-''-.   #
#   | ,_|     .'  .-,  '.  |   /  |   | .'_ _   \           /   _   \            \  _(`)_ \  .'_ _   \  .'  __ `.    /   __  \   .'_ _   \  #
# ,-./  )    / ,-.|  \ _ \ |  |   |  .'/ ( ` )   '         |  .` '\__|           | (_ o._)| / ( ` )   '/   '  \  \  | ,_/  \__) / ( ` )   ' #
# \  '_ '`) ;  \  '_ /  | :|  | _ |  |. (_ o _)  |          \  `--.              |  (_,_) /. (_ o _)  ||___|  /  |,-./  )      . (_ o _)  | #
#  > (_)  ) |  _`,/ \ _/  ||  _( )_  ||  (_,_)___|         /' ..--`.----.        |   '-.-' |  (_,_)___|   _.-`   |\  '_ '`)    |  (_,_)___| #
# (  .  .-' : (  '\_/ \   ;\ (_ o._) /'  \   .---.        :  `     ' ___|        |   |     '  \   .---..'   _    | > (_)  )  __'  \   .---. #
#  `-'`-'|___\ `"/  \  ) /  \ (_,_) /  \  `-'    /        |   `..-'_( )_         |   |      \  `-'    /|  _( )_  |(  .  .-'_/  )\  `-'    / #
#   |        \'. \_/``".'    \     /    \       /          \      (_ o _)        /   )       \       / \ (_ o _) / `-'`-'     /  \       /  #
#   `--------`  '-----'       `---`      `'-..-'            `-.__..(_,_)         `---'        `'-..-'   '.(_,_).'    `._____.'    `'-..-'   #
# ----------------------------------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser(description='Convert OpenSfM reconstruction to BlockExchange XML format.')
  parser.add_argument('--opensfm_dir', help='Directory containing OpenSfM output files (reconstruction.json, tracks.csv)')
  parser.add_argument('--filter', action='store_true', help='Apply filtering to tie points')
  args = parser.parse_args()
  
  # parse arguments
  directory_opensfm = args.opensfm_dir
  flag_filter = args.filter

  # set paths
  reconstruction_json_path = os.path.join(directory_opensfm , 'reconstruction.json')
  tracks_csv_path = os.path.join(directory_opensfm , 'converted_tracks.csv')

  if flag_filter:
    blocksexchange_xml_path = os.path.join(directory_opensfm , 'filter_AT.xml')
  else:
    blocksexchange_xml_path = os.path.join(directory_opensfm , 'blockexchangeAT.xml')

  metadeata_xml_path = os.path.join(directory_opensfm , 'metadata.xml')
  
  # read in reconstruction.json
  with open(reconstruction_json_path, 'r') as reconstruction_file:
    reconstruction_data = json.load(reconstruction_file)[0] # it's [{things}]

  origin = reconstruction_data['reference_lla']
  if origin['latitude'] == 0. and origin['longitude'] == 0. and origin['altitude'] == 0.:
    ori_utm = np.array([0., 0., 0.])
  else:
    ori_utm = convert_origin(origin)
  # convert origin to utm
  # get_metadata_xml(ori_utm, metadeata_xml_path)

  tracks_data = {}
  with open(tracks_csv_path, 'r') as tracks_file:
    tracks_file.readline() # skip the title line
    for line in tracks_file:
      str_list = line.strip().split('\t')
      if len(str_list) < 2: # skip the title line and the empty lines
        continue
      pic_name = str_list[0]
      track_id = str_list[1]
      u = str_list[3]
      v = str_list[4]
      if track_id in tracks_data:
        tracks_data[track_id].append([pic_name, u, v])
      else:
        tracks_data[track_id] = [[pic_name, u, v]]

  # init xml root
  root = init_root(ori_utm)

  # construct the photogroups part
  cam_to_phg = {}
  for i, (cam_name, cam_data) in enumerate(reconstruction_data['cameras'].items()):
    phg_name = 'Photogroup ' + str(i + 1)
    cam_to_phg[cam_name] = phg_name
    add_photogroup(root, cam_data, phg_name)



  # construct the photos part
  shots = reconstruction_data['shots']
  shots = dict(sorted(shots.items()))

  photo_data = convert_photo_data(shots, ori_utm)
  for pic_name, pic_data in photo_data.items():
    add_photo(root, pic_data, cam_to_phg[pic_data['Camera']])

  # construct the tie points part
  points = reconstruction_data['points']
  if flag_filter:
    tiepoints_data = convert_and_filter_tiepoints_data(points, tracks_data, photo_data, reconstruction_data['cameras'], ori_utm, 1.0, 2)
  else:
    tiepoints_data = convert_tiepoints_data(points, tracks_data, photo_data, reconstruction_data['cameras'], ori_utm)
  for tp_id, tp_data in tiepoints_data.items():
    add_tiepoint(root, tp_data)

  blocksexchange_data = ET.tostring(root, encoding='utf-8')
  blocksexchange_data = minidom.parseString(blocksexchange_data).toprettyxml(indent='\t')

  if os.path.exists(blocksexchange_xml_path):
    # 显式清空文件内容
    with open(blocksexchange_xml_path, 'w') as f:
        f.truncate(0)
  with open(blocksexchange_xml_path, 'w') as blocksexchange_file:
    blocksexchange_file.write(blocksexchange_data)
