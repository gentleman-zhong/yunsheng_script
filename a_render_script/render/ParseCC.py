#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------------------
# author    : You Zhou
# email     : zhouyou@ikingtec.com
# date      : Jul 22, 2025
# version   : 1.0
# brief     : Implementation of a series of methods that deal with CC xml blocks
# -------------------------------------------------------------------------------------------------
# history
# 250722    : v1.0, project created
# -------------------------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class CCBlockLoader:
  def __init__(self, _xml_file):
    self.xml_file        = _xml_file
    self.photogroup_dict = None      # {'NameList':['PHG1','PHG2'], 'EPSG':32650, 'PHG1':{'Camera':{}, 'Photos':{}}, 'PHG2':{}}
    self.tiepoint_list   = None      # [{}, {}, {}]
    self.Initialize()

  def Initialize(self):
    self.photogroup_dict, self.tiepoint_list = self.ParseCCBlock(self.xml_file)



  def ParseCCBlock(self, _xml_file):
    tree = ET.parse(_xml_file)
    root = tree.getroot()

    # define helper function ---------------------------------------------------\
    def extract_epsg(s):
      match = re.search(r'EPSG:(\d+)', s)
      return int(match.group(1)) if match else None
    # end def ------------------------------------------------------------------/
    # Parse Spatial Reference Systems element
    spatial_reference_systems_elem = root.find('SpatialReferenceSystems')
    if spatial_reference_systems_elem != None:
      # Initialize empty dictionaries
      srs_dict = {} # {ID: EPSG code}
      # loop and parse
      for srs_elem in spatial_reference_systems_elem.findall('SRS'):
        srs_id = int(srs_elem.find('Id').text)
        epsg = extract_epsg(srs_elem.find('Name').text) # could be None
        srs_dict[srs_id] = epsg
    else:
      srs_dict = {0: 0} # 0 be relative


    # Parse Block element
    block_elem = root.find('Block')
    block_srs_id = int(block_elem.find('SRSId').text) if block_elem.find('SRSId') != None else 0
    block_srs_epsg = srs_dict[block_srs_id] # 0 be relative
    # Initialize empty dictionaries
    photogroup_dict = {'NameList': [], 'EPSG': block_srs_epsg} # block srs is the srs of all photogroups in this block
    # Parse Photogroup element
    for photogroup_elem in block_elem.find('Photogroups').findall('Photogroup'):
      photogroup_name = photogroup_elem.find('Name').text
      photogroup_dict['NameList'].append(photogroup_name)
      photogroup_dict[photogroup_name] = {}
      photogroup_dict[photogroup_name]['Camera'] = {
        'Width': int(photogroup_elem.find('ImageDimensions').find('Width').text),
        'Height': int(photogroup_elem.find('ImageDimensions').find('Height').text),
        'PrincipalX': float(photogroup_elem.find('PrincipalPoint').find('x').text),
        'PrincipalY': float(photogroup_elem.find('PrincipalPoint').find('y').text),
        'K1': float(photogroup_elem.find('Distortion').find('K1').text),
        'K2': float(photogroup_elem.find('Distortion').find('K2').text),
        'K3': float(photogroup_elem.find('Distortion').find('K3').text),
        'P1': float(photogroup_elem.find('Distortion').find('P1').text),
        'P2': float(photogroup_elem.find('Distortion').find('P2').text)
      }
      if photogroup_elem.find('FocalLength') != None and photogroup_elem.find('SensorSize') != None:
        f = float(photogroup_elem.find('FocalLength').text)
        s = float(photogroup_elem.find('SensorSize').text)
        photogroup_dict[photogroup_name]['Camera']['FocalPixels'] = photogroup_dict[photogroup_name]['Camera']['Width'] * f / s
        print('FocalPixels:', photogroup_dict[photogroup_name]['Camera']['FocalPixels'])
      elif photogroup_elem.find('FocalLengthPixels') != None:
        photogroup_dict[photogroup_name]['Camera']['FocalPixels'] = float(photogroup_elem.find('FocalLengthPixels').text)
      else:
        print('Error: focal is not given!')

      # Parse Photo element
      photo_dict = {}
      for elem_per_photogroup in photogroup_elem:
        if elem_per_photogroup.tag != 'Photo':
          continue
        component = int(elem_per_photogroup.find('Component').text)
        if component == 0: # photos that are not connected
          continue
        key = int(elem_per_photogroup.find('Id').text)
        photo_dict[key] = {'R':[], 'T': []}
        photo_dict[key]['Name'] = elem_per_photogroup.find('ImagePath').text.strip().split("/")[-1]
        for elem_per_rotation in elem_per_photogroup.find('Pose').find('Rotation'):
          if elem_per_rotation.tag != 'Accurate':
            photo_dict[key]['R'].append(float(elem_per_rotation.text))
        for elem_per_center in elem_per_photogroup.find('Pose').find('Center'):
          if elem_per_center.tag != 'Accurate':
            photo_dict[key]['T'].append(float(elem_per_center.text))
      photogroup_dict[photogroup_name]['Photos'] = photo_dict


    # parse tiepoints
    tiepoints_elem = block_elem.find('TiePoints')
    tiepoint_list = []
    if tiepoints_elem != None: # block.xml may not contain tiepoints, return empty list in this case
      for tiepoint_elem in tiepoints_elem:
        tiepoint_dic = {}
        coords = tiepoint_elem.find('Position')
        if coords == None:
          continue
        tiepoint_dic['Position'] = [float(coords.find('x').text), float(coords.find('y').text), float(coords.find('z').text)]
        color = tiepoint_elem.find('Color')
        tiepoint_dic['Color'] = [float(color.find('Red').text), float(color.find('Green').text), float(color.find('Blue').text)]
        tiepoint_list.append(tiepoint_dic)

    return photogroup_dict, tiepoint_list


  def GetRTList(self, _block_dic, _epsg = None, _srs_origin = None):
    # _epsg needs to be used
    phg_list = _block_dic['NameList']

    Rlist = []
    Tlist = []
    for phg_name in phg_list:
      phg_dic = _block_dic[phg_name]
      photo_dic = phg_dic['Photos']

      for v in photo_dic.values():
        R = np.array(v['R'])
        R = R.reshape(3,3)
        T = np.array(v['T']) - _srs_origin if _srs_origin != None else np.array(v['T'])
        RT = np.zeros((4, 4))
        RT[:3, :3] = R.T
        RT[:3, 3] = T
        RT[3, 3] = 1.0
        RT_inv = np.linalg.inv(RT)
        Rnew = RT_inv[:3, :3]
        Tnew = RT_inv[:3, 3]
        Rlist.append(Rnew)
        Tlist.append(Tnew)
    Rlist = np.array(Rlist)
    Tlist = np.array(Tlist)
    return Rlist, Tlist
  

  def GetRTDict(self, _block_dic, _epsg = None, _srs_origin = None):
    # _epsg needs to be used
    phg_list = _block_dic['NameList']

    Rdict = {}
    Tdict = {}
    for phg_name in phg_list:
      phg_dic = _block_dic[phg_name]
      photo_dic = phg_dic['Photos']
      Rdict[phg_name] = []
      Tdict[phg_name] = []

      for v in photo_dic.values():
        R = np.array(v['R'])
        R = R.reshape(3,3)
        # T = np.array(v['T']) - _srs_origin if _srs_origin != None else np.array(v['T'])
        if _srs_origin is not None:
          T = np.array(v['T']) - _srs_origin
        else:
          T = np.array(v['T'])
        RT = np.zeros((4, 4))
        RT[:3, :3] = R.T
        RT[:3, 3] = T
        RT[3, 3] = 1.0
        RT_inv = np.linalg.inv(RT)
        Rnew = RT_inv[:3, :3]
        Tnew = RT_inv[:3, 3]
        Rdict[phg_name].append(Rnew)
        Tdict[phg_name].append(Tnew)
      Rdict[phg_name] = np.array(Rdict[phg_name])
      Tdict[phg_name] = np.array(Tdict[phg_name])
    return Rdict, Tdict


  def GetPC(self, _tiepoint_list):
    ptlist = []
    colorlist = []
    for pt in _tiepoint_list:
      xyz = pt['Position']
      rgb = [round(c * 255) for c in pt['Color']]
      ptlist.append(xyz)
      colorlist.append(rgb)
    ptlist = np.array(ptlist)
    colorlist = np.array(colorlist)
    return ptlist, colorlist


  def PlotCameraAndPC(self, Rlist, Tlist, ptlist, colorlist):
    center = ptlist.mean(axis = 0)
    min_vals = ptlist.min(axis = 0)
    max_vals = ptlist.max(axis = 0)
    extent = max_vals - min_vals
    max_range = extent.max() / 5

    pt0_list = []
    pt1_list = []
    pt0 = np.array([0, 0, 0, 1])
    pt1 = np.array([0, 0, max_range / 20, 1])
    for i in range(len(Rlist)):
      R = Rlist[i]
      T = Tlist[i]
      Rt = np.zeros((4, 4))
      Rt[:3, :3] = R
      Rt[:3, 3] = T
      Rt[3, 3] = 1.0
      Rt_inv = np.linalg.inv(Rt)
      pt0_list.append(np.dot(Rt_inv, pt0))
      pt1_list.append(np.dot(Rt_inv, pt1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    for i in range(len(pt0_list)):
      dirt = pt1_list[i]- pt0_list[i]
      ax.quiver(pt0_list[i][0], pt0_list[i][1], pt0_list[i][2], dirt[0], dirt[1], dirt[2], color = 'black')
    colors = [mcolors.rgb2hex(c/255.) for c in colorlist]
    ax.scatter(ptlist[:, 0], ptlist[:, 1], ptlist[:, 2], s = 1, color = colors)
    xlim = (center[0] - max_range/2, center[0] + max_range/2)
    ylim = (center[1] - max_range/2, center[1] + max_range/2)
    zlim = (center[2] - max_range/2, center[2] + max_range/2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.show()