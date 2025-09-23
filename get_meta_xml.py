from pyproj import CRS
from pyproj import Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
import numpy as np
import argparse
import os
import json
import xml.etree.ElementTree as ET

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert OpenSfM reconstruction to BlockExchange XML format.')
  parser.add_argument('--opensfm_dir', help='Directory containing OpenSfM output files (reference_lla.json)')
  args = parser.parse_args()
  reference_lla_json_path = os.path.join(args.opensfm_dir,'reference_lla.json')
  with open(reference_lla_json_path, 'r') as f:
    reconstruction_lla_data = json.load(f)
  ori_utm = convert_origin(reconstruction_lla_data)

  metadeata_xml_path = os.path.join(args.opensfm_dir , 'metadata.xml')
  get_metadata_xml(ori_utm, metadeata_xml_path)