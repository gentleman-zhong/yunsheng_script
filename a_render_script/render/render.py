import blenderproc as bproc
import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
import re
import argparse
# import debugpy
import glob

sys.path.append(os.getcwd())
from ParseCC import CCBlockLoader



def parse_obj_metadata(_xml_file):
  tree = ET.parse(_xml_file)
  root = tree.getroot()

  # define helper function ---------------------------------------------------\
  def extract_epsg(s):
    match = re.search(r'EPSG:(\d+)', s)
    return int(match.group(1)) if match else None
  # end def ------------------------------------------------------------------/
  def extract_origin(s):
    return np.array(s.split(','), dtype = float)

  srs_elem = root.find('SRS')
  srs_origin_elem = root.find('SRSOrigin')
  if srs_elem != None and srs_origin_elem!= None:
    epsg = extract_epsg(srs_elem.text)
    srs_origin = extract_origin(srs_origin_elem.text)
    return epsg, srs_origin
  else:
    return None, None



def render(_input_folder, _output_folder, _block, _epsg, _srs_origin):
  # initialize the scene
  bproc.init()
  # load obj files
  objs = []
  for dirpath, dirnames, filenames in os.walk(_input_folder):
    # print(dirpath, dirnames, filenames)
    for filename in filenames:
        if filename.lower().endswith('.obj'):
            obj_path = os.path.join(dirpath, filename)
            print(f'Importing: {obj_path}')
            objs += bproc.loader.load_obj(obj_path)
  for obj in objs:
    # set the axis direction correct
    obj.set_rotation_euler([0., 0., 0.])
    for mat in obj.get_materials():
        # Create a new emission material, ignores lighting and shows base color
        mat.make_emissive(emission_strength=1.0, replace=True)

  block_dic = _block.photogroup_dict
  phg_list = block_dic['NameList']
  R_dic, T_dic = _block.GetRTDict(block_dic, _epsg, _srs_origin)
  for phg_name in phg_list:
    phg_dic = block_dic[phg_name]
    camera_dic = phg_dic['Camera']
    Rlist = R_dic[phg_name]
    Tlist = T_dic[phg_name]

    w = camera_dic['Width']
    h = camera_dic['Height']
    fx = camera_dic['FocalPixels']
    fy = camera_dic['FocalPixels']
    cx = camera_dic['PrincipalX']
    cy = camera_dic['PrincipalY']
    K = np.array([
      [fx, 0, cx], 
      [0, fy, cy], 
      [0, 0, 1]
      ])
    clip_start = 0.05
    clip_end = 1000.0
    bproc.camera.set_intrinsics_from_K_matrix(K, w, h, clip_start, clip_end)
    k1 = camera_dic['K1']
    k2 = camera_dic['K2']
    k3 = camera_dic['K3']
    p1 = camera_dic['P1']
    p2 = camera_dic['P2']
    mapping = bproc.camera.set_lens_distortion(k1, k2, k3, p1, p2)

    # used to convert blender coords system to camera system
    Rx = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    for i in range(len(Rlist)):
      if i % 3 == 0:
        R = Rlist[i]
        T = Tlist[i]
        print(i,f'Adding camera pose: {R}, {T}')
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        Rt_inv = np.linalg.inv(Rt)
        bproc.camera.add_camera_pose(Rt_inv @ Rx)
       # only use the first camera pose for now
  # end for
  # bproc.renderer.set_device("GPU")

  bproc.renderer.set_max_amount_of_samples(256)
  bproc.renderer.set_output_format(enable_transparency=True)
  data = bproc.renderer.render()
  data['colors'] = bproc.postprocessing.apply_lens_distortion(data['colors'], mapping, w, h, use_interpolation=True)
  bproc.writer.write_hdf5(_output_folder, data)


def main():
    parser = argparse.ArgumentParser(description="Render HDF5 from 3D model with camera positions.")
    parser.add_argument('--obj_folder', type=str, required=True, help='Path to input model folder')
    parser.add_argument('--xml_dir', type=str, required=True, help='Path to xml (camera poses)')

    args = parser.parse_args()

    obj_folder = args.obj_folder
    output_folder = os.path.join(args.obj_folder,'render_output')
    print(output_folder)
    cc_file = glob.glob(os.path.join(args.xml_dir, '*AT.xml'))[0]
    meta_file = os.path.join(args.xml_dir, 'metadata.xml')
# parse metadata
    if os.path.exists(meta_file):
      epsg, srs_origin = parse_obj_metadata(meta_file)
    else:
      epsg, srs_origin = None, None
    print(f'epsg: {epsg}, srs_origin: {srs_origin}')
    block = CCBlockLoader(cc_file)
    # print(block.photogroup_dict)
    render(obj_folder, output_folder, block, epsg, srs_origin )


if __name__ == "__main__":
    main()