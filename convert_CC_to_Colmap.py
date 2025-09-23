# By You, converts ContextCapture blocks into Colmap txt formats
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

def parse_block_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize empty dictionaries
    photogroup_dict = {}
    photo_dict = {}

    # Parse Block element
    block_elem = root.find('Block')

    # Parse Photogroup element
    photogroup_elem = block_elem.find('Photogroups').find('Photogroup')
    photogroup_dict['CameraParams'] = {
        'Width': int(photogroup_elem.find('ImageDimensions').find('Width').text),
        'Height': int(photogroup_elem.find('ImageDimensions').find('Height').text),
        'Focal': float(photogroup_elem.find('Photo').find('ExifData').find('FocalLength').text),
        'FocalPixels': float(photogroup_elem.find('FocalLengthPixels').text),
        'PrincipalX': float(photogroup_elem.find('PrincipalPoint').find('x').text),
        'PrincipalY': float(photogroup_elem.find('PrincipalPoint').find('y').text)
    }

    # Parse Photo element
    for elem_per_photogroup in photogroup_elem:
        if elem_per_photogroup.tag != 'Photo':
            continue
        component = int(elem_per_photogroup.find('Component').text)
        if component == 0:
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

    # Combine dictionaries into a final result
    photogroup_dict['Photo'] = photo_dict

    tiepoints_elem = block_elem.find('TiePoints')
    tiepoint_list = []
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

def get_quaternions(R):
    """
    Convert a 3x3 rotation matrix to a unit quaternion.
    
    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix

    Returns:
        numpy.ndarray: Unit quaternion [w, x, z]
    """
    # Extract rotation matrix elements for clarity
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    
    # Calculate quaternion components
    qw = np.sqrt(1.0 + m00 + m11 + m22) / 2.0
    qx = (m21 - m12) / (4.0 * qw)
    qy = (m02 - m20) / (4.0 * qw)
    qz = (m10 - m01) / (4.0 * qw)
    
    return np.array([qw, qx, qy, qz])

def plot_camera_and_pc(Rlist, Tlist, ptlist, colorlist):
    pt0_list = []
    pt1_list = []
    pt0 = np.array([0, 0, 0, 1])
    pt1 = np.array([0, 0, 1, 1])
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
    ax.scatter(ptlist[:, 0], ptlist[:, 1], ptlist[:, 2], color = colors)
    plt.show()



cc_file = sys.argv[1]
pos_dict, tiepoint_list = parse_block_to_dict(cc_file)
camera_dic = pos_dict['CameraParams']
photo_dic = pos_dict['Photo']

Rlist = []
Tlist = []
ptlist = []
colorlist = []

with open('cameras.txt', 'w') as outputfile:
    outputfile.write('# Camera list with one line of data per camera:\n')
    outputfile.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
    outputfile.write('# Number of cameras: 1\n')
    outputfile.write('1 PINHOLE '+str(camera_dic['Width'])+' '\
                     +str(camera_dic['Height'])+' '+str(camera_dic['FocalPixels'])+' '\
                        +str(camera_dic['FocalPixels'])+' '+str(camera_dic['PrincipalX'])+' '\
                            +str(camera_dic['PrincipalY']))
    
with open('images.txt', 'w') as outputfile:
    outputfile.write('# Image list with two lines of data per image:\n')
    outputfile.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
    outputfile.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
    n = len(photo_dic)
    mean_obs = 0.0
    outputfile.write('# Number of images: '+str(n)+', mean observations per image: '+str(mean_obs)+'\n')
    for k,v in photo_dic.items():
        R = np.array(v['R'])
        R = R.reshape(3,3)
        T = T = np.array(v['T'])
        RT = np.zeros((4, 4))
        RT[:3, :3] = R.T
        RT[:3, 3] = T
        RT[3, 3] = 1.0
        RT_inv = np.linalg.inv(RT)
        Rnew = RT_inv[:3, :3]
        Tnew = RT_inv[:3, 3]
        Rlist.append(Rnew)
        Tlist.append(Tnew)
        Q = get_quaternions(Rnew)
        line_str = str(k)+' '+str(Q[0])+' '+str(Q[1])+' '+str(Q[2])+' '+str(Q[3])+' '\
                                    +str(Tnew[0])+' '+str(Tnew[1])+' '+str(Tnew[2])+' 1 '+v['Name']+'\n'
        outputfile.write(line_str)
        outputfile.write('\n')

with open('points3D.txt', 'w') as outputfile:
    outputfile.write('# 3D point list with one line of data per point:\n')
    outputfile.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
    mean_trk = 0.0
    outputfile.write('# Number of points: '+str(n)+', mean track length: '+str(mean_trk)+'\n')
    for i,pt in enumerate(tiepoint_list):
        xyz = pt['Position']
        rgb = [round(c * 255) for c in pt['Color']]
        ptlist.append(xyz)
        colorlist.append(rgb)
        line_str = str(i)+' '+str(xyz[0])+' '+str(xyz[1])+' '+str(xyz[2])+' '\
                                    +str(rgb[0])+' '+str(rgb[1])+' '+str(rgb[2])+' 0.0 0 0\n'
        outputfile.write(line_str)

Rlist = np.array(Rlist)
Tlist = np.array(Tlist)
ptlist = np.array(ptlist)
colorlist = np.array(colorlist)
if sys.argv[2] == 'True':
    plot_camera_and_pc(Rlist, Tlist, ptlist, colorlist)