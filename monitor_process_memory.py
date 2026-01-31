import psutil
import time
import os

# --- 配置区域 ---
MAIN_SCRIPT = "opensfm_main.py"
SUB_COMMANDS = ["post_processing_ba", "reconstruct"] # 满足其中一个即可
LOG_NAME = "memory_usage_detail.log"
INTERVAL = 5  # 采样频率（秒）

def get_dynamic_log_path(cmdline_list):
    """
    从命令行参数中提取任务路径
    """
    if not cmdline_list or len(cmdline_list) < 2:
        return None
        
    # 通常路径是命令行中的最后一个参数
    last_arg = cmdline_list[-1]
    
    # 提取该路径所在的文件夹作为日志存放点
    if os.path.exists(last_arg):
        return os.path.dirname(last_arg)
    elif "/" in last_arg: 
        return os.path.dirname(last_arg)
    return None

def main():
    print(f"正在启动动态路径监测脚本...")
    print(f"监测脚本: {MAIN_SCRIPT}")
    print(f"监测子命令: {SUB_COMMANDS}")
    print(f"数据单位: GB (对应 htop 的 VIRT 和 RES)")
    print("---------------------------------------")

    try:
        while True:
            found_procs = []
            # 1. 寻找匹配进程
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if not cmdline: continue
                    
                    cmd_str = " ".join(cmdline)
                    
                    # --- 修改后的逻辑 ---
                    # 必须包含 opensfm_main.py
                    if MAIN_SCRIPT in cmd_str:
                        # 并且包含 reconstruct 或 post_processing_ba 中的任意一个
                        if any(sub in cmd_str for sub in SUB_COMMANDS):
                            found_procs.append(proc)
                    # --------------------
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 2. 逐个记录信息
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            if found_procs:
                for proc in found_procs:
                    try:
                        # 动态解析日志路径
                        cmdline_list = proc.info['cmdline']
                        target_dir = get_dynamic_log_path(cmdline_list)
                        if not target_dir:
                            continue
                        
                        log_path = os.path.join(target_dir, LOG_NAME)
                        
                        # 如果是新日志，写表头
                        if not os.path.exists(log_path):
                            with open(log_path, "w", encoding="utf-8") as f:
                                f.write("Time\tPID\tVIRT_GB\tRES_GB\tCPU%\n")

                        # 获取内存信息
                        mem_info = proc.memory_info()
                        virt_gb = mem_info.vms / (1024 ** 3)
                        res_gb = mem_info.rss / (1024 ** 3)
                        cpu = proc.cpu_percent(interval=None)

                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"{current_time}\t{proc.pid}\t{virt_gb:.2f}\t{res_gb:.2f}\t{cpu}\n")
                        
                        print(f"[{current_time}] PID:{proc.pid} | VIRT:{virt_gb:.1f}G | RES:{res_gb:.1f}G | 日志:{log_path}")
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                        continue
            
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n监测已停止。")

if __name__ == "__main__":
    main()
