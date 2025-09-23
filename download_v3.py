import os
import requests
import time
import tkinter as tk
from tkinter import filedialog

# Define a function to download images with retries
def download_image(url, save_folder, timeout=5, retries=10):
    # Extract the image name from the URL
    image_name = url.split("/")[-1]
    
    # 创建子文件夹（使用文件名前6位）
    if len(image_name) >= 6:
        sub_folder = image_name[:6]
        folder_path = os.path.join(save_folder, sub_folder)
        create_directory(folder_path)
        image_path = os.path.join(folder_path, image_name)
    else:
        image_path = os.path.join(save_folder, image_name)

    # Check if the image already exists
    if os.path.exists(image_path):
        print(f"Image {image_name} already exists, skipping download.")
        return

    # Retry logic
    attempt = 0
    while attempt < retries:
        try:
            # Get the image content from the URL with a timeout
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()  # Raise an error if the request was unsuccessful
            
            # Save the image to the specified directory
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            print(f"Downloaded {image_name}")
            return  # Successfully downloaded, exit the function
        
        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Failed to download {url} (Attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                print(f"Retrying in 1 seconds...")
                time.sleep(1)  # Wait for a few seconds before retrying
            else:
                print(f"Giving up on {image_name} after {retries} attempts.")

# Main function to download all images from a file
def download_images_from_file(file_path, save_folder, timeout=10, retries=3):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    counter = 0
    with open(file_path, 'r') as file:
        # Read each line (image URL) from the file
        for line in file:
            url = line.strip()
            if url:
                counter += 1
                print(counter, '\t', end = '')
                download_image(url, save_folder, timeout, retries)

def download_txt_from_url(route_id):
    """
    从指定URL下载txt文件
    """
    url = f"https://api.ikingtec.com/station-openApi/device/v1/open/shanhaijing/images?routeId={route_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # 创建临时txt文件保存URL列表
        temp_file = f"route_{route_id}.txt"
        with open(temp_file, 'w') as f:
            f.write(response.text)
        return temp_file
    except requests.exceptions.RequestException as e:
        print(f"下载URL列表失败: {str(e)}")
        return None

def create_directory(directory):
    """
    安全地创建目录
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录 {directory} 失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 创建一个隐藏的主窗口
    root = tk.Tk()
    root.withdraw()

    
    route_id = input("请输入航线编号（输入q退出）: ")
    if route_id.lower() == 'q':
        print("程序结束")
        exit()
        
    # 下载txt文件
    txt_file_path = download_txt_from_url(route_id)
    if not txt_file_path:
        print("获取URL列表失败，请重试")
        exit()
        

    # 打开文件夹选择对话框
    save_folder = filedialog.askdirectory(title="选择保存图片的文件夹")
    if not save_folder:
        print("未选择保存文件夹，使用默认文件夹'未命名'")
        save_folder = '未命名'

    
    # 开始下载图片
    download_images_from_file(txt_file_path, save_folder)
    
    # 删除临时txt文件
    try:
        os.remove(txt_file_path)
    except OSError as e:
        print(f"删除临时文件失败: {str(e)}")