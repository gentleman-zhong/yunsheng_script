import sys
from ParseCC import CCBlockLoader
import matplotlib.pyplot as plt

cc_file = sys.argv[1]
flag_plot = sys.argv[2] == 'True'

block = CCBlockLoader(cc_file)
block.Initialize()

blk_dic = block.photogroup_dict
tpt_list = block.tiepoint_list

Rlist, Tlist = block.GetRTList(blk_dic)
ptlist, colorlist = block.GetPC(tpt_list)

if flag_plot:
    block.PlotCameraAndPC(Rlist, Tlist, ptlist, colorlist)
    fig = plt.gcf()  # 获取当前 figure
    output_path = "output.png"
    fig.savefig(output_path)
    print(f"图像已保存到 {output_path}")