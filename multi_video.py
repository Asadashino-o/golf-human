import argparse
import time
import torch
from head import head
from head_and_buttock import main
import multiprocessing


def worker(gpu_id, input, output, idx, interval, category):
    # 设置指定GPU
    torch.cuda.set_device(gpu_id)
    if category == 1:
        main(input, output, idx, interval, None)
    else:
        head(input, output, idx, interval, None)
    # 在视频上执行推理
    # 这里需要根据具体情况编写视频处理的代码
    print(f"Processing {input} on GPU {gpu_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fi", "--input_video_path0", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-fo", "--output_video_path0", type=str, default="./output_video.mp4"
    )
    parser.add_argument("-ff", "--f_idx0", type=int)
    parser.add_argument(
        "-di", "--input_video_path1", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-do", "--output_video_path1", type=str, default="./output_video.mp4"
    )
    parser.add_argument("-df", "--f_idx1", type=int)
    parser.add_argument("-n", "--interval", type=int)
    args = parser.parse_args()

    # GPU数量
    num_gpus = torch.cuda.device_count()

    # 启动多个进程
    processes = []
    # 记录开始时间
    start_time = time.time()
    for i in range(2):  # 循环次数应该是2，不是范围为0到2
        gpu_id = i % num_gpus  # 确保每个视频对应不同的GPU
        # 使用字符串格式化来获取args中对应编号的视频路径
        input_video_path = getattr(args, f'input_video_path{i}')
        output_video_path = getattr(args, f'output_video_path{i}')
        f_idx = getattr(args, f'f_idx{i}')
        p = multiprocessing.Process(target=worker,
                                    args=(gpu_id, input_video_path, output_video_path, f_idx, args.interval, i))
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    runtime = end_time - start_time
    print("程序运行时间为：", runtime, "秒")
