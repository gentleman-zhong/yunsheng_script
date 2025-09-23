#!/bin/bash
    # "/datasets/code/six_datasets_shanhaijing/250718_feilou"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing/250718_dongman"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing/250718_gongsi"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing/250718_guanyin"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing/250718_tianfu"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing/250718_houzi"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing/250718_feilou"
# 所有数据集路径
datasets=(
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_tianfu"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_dongman"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_houzi"
    # "/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_gongsi"
    "/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/250718_guanyin"
)

# 日志输出目录
log_dir="/home/zhangzhong/experiment/six_datasets_shanhaijing_CC_2_ODM/run_all_datasets_odm_logs"

# 检查目录是否存在
if [ -d "$log_dir" ]; then
    echo "目录已存在，删除中..."
    rm -rf "$log_dir"  # 删除目录及其内容
    echo "目录已删除"
fi

# 重新创建目录
mkdir -p "$log_dir"
echo "目录已重新创建: $log_dir"


# 遍历数据集逐个执行，并保存日志
for dataset in "${datasets[@]}"; do
    name=$(basename "$dataset")
    echo ">>>> 开始运行数据集: $name"

    log_file="${log_dir}/${name}.log"

    {
        echo "开始运行: $dataset"
        date

        # 执行主流程
        bash /code/run.sh "$dataset" --my-stage 0
        # bash /code/run.sh "$dataset" --my-stage 1
        # bash /code/run.sh "$dataset" --my-stage 2
        # bash /code/run.sh "$dataset" --my-stage 3
        # bash /code/run.sh "$dataset" --my-stage 4
        # bash /code/run.sh "$dataset" --my-stage 5
        # bash /code/run.sh "$dataset" --my-stage 6
        # bash /code/run.sh "$dataset" --my-stage 7
        # bash /code/run.sh "$dataset" --my-stage 8
        # bash /code/run.sh "$dataset" --my-stage 9
        # bash /code/run.sh "$dataset" --my-stage 10
        # bash /code/run.sh "$dataset" --my-stage 11
        # bash /code/run.sh "$dataset" --my-stage 12


        status=$?
        echo "返回值: $status"
        date

        if [ $status -ne 0 ]; then
            echo "!!! 运行失败: $dataset"
        else
            echo "=== 运行成功: $dataset"
        fi
    } | tee "$log_file"

    echo ">>>> 日志已保存至: $log_file"
    echo ""
done
