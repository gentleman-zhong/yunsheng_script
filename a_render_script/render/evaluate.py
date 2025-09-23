import os
import subprocess
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import argparse

def save_png(folder_hdf5, folder_png):
    files_hdf5 = sorted([f for f in os.listdir(folder_hdf5) if f.lower().endswith(('.hdf5'))], key = lambda x: int(x.split('.')[0]))
    for f in files_hdf5:
        fpath = os.path.join(folder_hdf5, f)
        cmd = ['blenderproc', 'vis', 'hdf5', fpath, '--save', folder_png]
        subprocess.run(cmd, check=True)

def compute_metrics(folder_ori, folder_png):

    if folder_ori.split('/')[-2].endswith('only_odm'):
        folder_ori = os.path.join(os.path.dirname(os.path.dirname(folder_ori)), '250718_', folder_ori.split('/')[-2].split('_')[1], 'images')
    if folder_ori.split('/')[-2] == '250718_gongsi':
        files_ori = sorted([f for f in os.listdir(folder_ori) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key = lambda x: int(x.split('_')[-1].split('.')[0]))
    else:
        files_ori = sorted([f for f in os.listdir(folder_ori) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
  
    files_png = sorted([f for f in os.listdir(folder_png) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key = lambda x: int(x.split('_')[0]))
    
    files_ori_3 = []
    for idx, item in enumerate(files_ori):
        if idx % 3 == 0:
            files_ori_3.append(item) 

    if len(files_ori_3) != len(files_png):
        raise ValueError("Folders contain different number of images")

    psnr_list = []
    ssim_list = []

    for idx, (f1, f2) in enumerate(zip(files_ori_3, files_png)):
        # if idx  >= 10:
        #     f1 = f"{f1[:4]}{int(f1[4:8])+1:04d}{f1[8:]}"
        path1 = os.path.join(folder_ori, f1)
        path2 = os.path.join(folder_png, f2)
        try:
            img1 = imread(path1)
            img2 = imread(path2)
            mask1 = np.ones((img1.shape[:2]), dtype=int)
            mask2 = np.ones((img2.shape[:2]), dtype=int)

            if img1.ndim == 3 and img1.shape[-1] == 4:
                alpha1 = img1[..., 3] / 225.0 if img1.dtype == np.uint8 else img1[..., 3]
                mask1 = alpha1 > 0.5
                img1 = img1[..., :3]
            if img2.ndim == 3 and img2.shape[-1] == 4:
                alpha2 = img2[..., 3] / 225.0 if img2.dtype == np.uint8 else img2[..., 3]
                mask2 = alpha2 > 0.5
                img2 = img2[..., :3]

            if img1.shape != img2.shape:
                raise ValueError(f"Image shape mismatch: {f1} vs {f2}")
            
            combined_mask = (mask1 & mask2).astype(bool)

            img1_masked = img1[combined_mask]
            img2_masked = img2[combined_mask]

            img1_blacked = img1.copy().astype(np.float32)
            img2_blacked = img2.copy().astype(np.float32)
            for c in range(3):
                img1_blacked[..., c][~combined_mask] = 0
                img2_blacked[..., c][~combined_mask] = 0

            psnr_val = psnr(img1_masked, img2_masked, data_range=img1_masked.max() - img1_masked.min())
            ssim_val = ssim(img1_blacked, img2_blacked, data_range=img1_blacked.max() - img1_blacked.min(), channel_axis=-1 if img1_blacked.ndim == 3 else None)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue


        print(f"{f1} {f2} | PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    print("\nAverage PSNR:", np.mean(psnr_list))
    print("Average SSIM:", np.mean(ssim_list))
    print()

def overlay_images(folder_ori, folder_png, folder_comp):
    os.makedirs(folder_comp, exist_ok=True)  # 确保输出目录存在
    if folder_ori.split('/')[-2].endswith('only_odm'):
        folder_ori = os.path.join(os.path.dirname(os.path.dirname(folder_ori)), '250718_', folder_ori.split('/')[-2].split('_')[1], 'images')
    if folder_ori.split('/')[-2] == '250718_gongsi':
        files_ori = sorted([f for f in os.listdir(folder_ori) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key = lambda x: int(x.split('_')[-1].split('.')[0]))
    else:
        files_ori = sorted([f for f in os.listdir(folder_ori) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    files_png = sorted([f for f in os.listdir(folder_png) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=lambda x: int(x.split('_')[0]))


    files_ori_3 = []
    for idx, item in enumerate(files_ori):
        if idx % 3 == 0:
            files_ori_3.append(item)     
    if len(files_ori_3) != len(files_png):
        raise ValueError("Folders contain different number of images")

    for idx, (f1, f2) in enumerate(zip(files_ori_3, files_png)):
        # if idx  >= 10:
        #     f1 = f"{f1[:4]}{int(f1[4:8])+1:04d}{f1[8:]}"
        print('comparing', f1, f2)
        path1 = os.path.join(folder_ori, f1)
        path2 = os.path.join(folder_png, f2)
        path3 = os.path.join(folder_comp, 'compare_' + f2.split('_')[0] + '.png')
        try:
            bottom_img = Image.open(path1).convert('RGBA')
            top_img = Image.open(path2).convert('RGBA')
        except Exception as e:
            print(f"Error reading image: {e}")
            continue

        alpha = 85
        top_img.putalpha(alpha)

        blended_img = Image.alpha_composite(bottom_img, top_img)
        blended_img.save(path3)

def test_metrics(path1, path2):
    img1 = imread(path1)
    img2 = imread(path2)
    mask1 = np.ones((img1.shape[:2]), dtype=int)
    mask2 = np.ones((img2.shape[:2]), dtype=int)

    if img1.ndim == 3 and img1.shape[-1] == 4:
        alpha1 = img1[..., 3] / 225.0 if img1.dtype == np.uint8 else img1[..., 3]
        mask1 = alpha1 > 0.5
        img1 = img1[..., :3]
    if img2.ndim == 3 and img2.shape[-1] == 4:
        alpha2 = img2[..., 3] / 225.0 if img2.dtype == np.uint8 else img2[..., 3]
        mask2 = alpha2 > 0.5
        img2 = img2[..., :3]

    if img1.shape != img2.shape:
        raise ValueError(f"Image shape mismatch: {f1} vs {f2}")
    
    combined_mask = (mask1 & mask2).astype(bool)

    img1_masked = img1[combined_mask]
    img2_masked = img2[combined_mask]

    img1_blacked = img1.copy().astype(np.float32)
    img2_blacked = img2.copy().astype(np.float32)
    for c in range(3):
        img1_blacked[..., c][~combined_mask] = 0
        img2_blacked[..., c][~combined_mask] = 0

    psnr_val = psnr(img1_masked, img2_masked, data_range=img1_masked.max() - img1_masked.min())
    ssim_val = ssim(img1_blacked, img2_blacked, data_range=img1_blacked.max() - img1_blacked.min(), channel_axis=-1 if img1_blacked.ndim == 3 else None)
    print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input model folder')
    args = parser.parse_args()

    folder_hdf5 = os.path.join(args.input_folder, 'render_output')
    changjing_name = args.input_folder.split('/')[-7]
    folder_ori = os.path.join('/home/zhangzhong/experiment/six_datasets_shanhaijing',f'250718_{changjing_name}', 'images')
    folder_png = os.path.join(args.input_folder, 'renders_png_output')
    folder_comp = os.path.join(args.input_folder, 'renders_compare')
    save_png(folder_hdf5, folder_png)
    compute_metrics(folder_ori, folder_png)
    overlay_images(folder_ori, folder_png, folder_comp)