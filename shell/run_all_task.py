import os
import concurrent.futures
import subprocess
from typing import List

def run_commands_parallel(commands: List[str], log_dir: str, num_gpus: int):
    """
    并行运行命令，任务最大并行数等于显卡数，确保每个任务独占一个显卡。

    参数：
    - commands: 命令列表
    - log_dir: 日志文件夹路径，会自动创建并清空旧日志
    - num_gpus: 可用显卡数量，且最大并行任务数
    """
    max_workers = num_gpus
    os.makedirs(log_dir, exist_ok=True)

    # 清空日志目录内容
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def run_command(index: int, cmd: str):
        gpu_id = index % num_gpus
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_path = os.path.join(log_dir, f"cmd_{index}.log")
        with open(log_path, "w") as log_file:
            log_file.write(f"\n=== 开始执行：{cmd} (使用GPU {gpu_id}) ===\n")
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=env
            )
            log_file.write(f"\n=== 结束：{cmd}，返回码：{result.returncode} ===\n")
        return result.returncode

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_command, i, cmd)
            for i, cmd in enumerate(commands)
        ]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()
