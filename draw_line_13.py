import os
import cv2
import time
import numpy as np
import torch
from head_and_buttock import initialize_detector,inference
from head_and_buttock import get_head,get_buttocks,draw_y_rectangle,draw_x_rectangle


def process_images(folder_path):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print("文件夹路径不存在")
        return
    # Process each frame in the video
    frame_num = 0
    begin_X1 = 0
    begin_Y1 = 0
    begin_X2 = 0
    begin_Y2 = 0
    x_left = 0
    x_right = 0
    y_up = 0
    y_down = 0
    prev_x1 = 0
    prev_y1 = 0
    prev_x2 = 0
    prev_y2 = 0
    width = 0
    height = 0

    # 按照文件名顺序读取并处理图片
    for i in range(0, 12):
        filename = f"{i}.jpg"
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否存在
        if not os.path.isfile(file_path):
            print(f"文件 {filename} 不存在")
            continue

        # 使用 cv2 读取图片
        img = cv2.imread(file_path)
        # Track the region to color
        colored_region = np.zeros_like(img)

        height, width, _ = img.shape
        predictor = initialize_detector()
        filtered_results = inference(img, predictor)
        if filtered_results is None:
            frame_num += 1
            print(f"frame {frame_num} can't be detected")
            continue
        for result, box in filtered_results:
            iuv_array = torch.cat(
                (result.labels[None].type(torch.float32), result.uv * 255.0)
            ).type(torch.uint8)
            iuv_array = iuv_array.cpu().numpy()  # 将 CUDA tensor 转换为 NumPy 数组
            head_coords = get_head(iuv_array, box)
            buttocks_coords = get_buttocks(iuv_array, box)
            if buttocks_coords is None:
                x1 = prev_x1
                y1 = prev_y1
            else:
                x1, y1 = buttocks_coords

            if head_coords is None:
                x2 = prev_x2
                y2 = prev_y2
            else:
                x2, y2 = head_coords

            prev_x1 = x1
            prev_y1 = y1
            prev_x2 = x2
            prev_y2 = y2
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if frame_num == 0:
                begin_X1 = x1
                begin_Y1 = y1
                begin_X2 = x2
                begin_Y2 = y2
                x_left = begin_X1
                x_right = begin_X1
                y_up = begin_Y2
                y_down = begin_Y2
                _, _, width, height = box

            x_left = min(x_left, x1)
            x_right = max(x_right, x1)
            y_up = min(y_up, y2)
            y_down = max(y_down, y2)

            cv2.line(img, (x1, y1 - 60), (x1, y1 + 60), (0, 0, 255), thickness=2)  # 在 y 坐标处画一条绿色的水平线
            cv2.line(img, (begin_X1, begin_Y1 - 100), (begin_X1, begin_Y1 + 100), (0, 255, 0), thickness=2)
            draw_x_rectangle(begin_Y1, begin_X1, x_right, colored_region, color=0)  # 画右边的距离
            draw_x_rectangle(begin_Y1, x_left, begin_X1, colored_region, color=1)  # 画左边的距离

            cv2.line(img, (x2 - 60, y2), (x2 + 60, y2), (0, 0, 255), thickness=2)  # 在 x 坐标处画一条红色的水平线
            cv2.line(img, (begin_X2 - 100, begin_Y2), (begin_X2 + 100, begin_Y2), (0, 255, 0), thickness=2)
            draw_y_rectangle(begin_X2, y_up, begin_Y2, colored_region, color=0)  # 画上面的距离
            draw_y_rectangle(begin_X2, begin_Y2, y_down, colored_region, color=1)  # 画下面的距离

            cv2.imwrite(f"13_frame/{frame_num}.jpg", cv2.addWeighted(img, 0.8, colored_region, 0.2, 0))

            frame_num += 1
            print(f"Processed frame {frame_num} / 12")

    dis_up = (begin_Y2 - y_up) / height
    dis_down = (y_down - begin_Y2) / height
    dis_left = (begin_X1 - x_left) / width
    dis_right = (x_right - begin_X1) / width
    print("臀部向左相对移动距离：{}".format(dis_left))
    print("臀部向右相对移动距离：{}".format(dis_right))
    print("头部向上相对移动距离：{}".format(dis_up))
    print("头部向下相对移动距离：{}".format(dis_down))


if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ")
    # 记录开始时间
    start_time = time.time()

    process_images(folder_path)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    runtime = end_time - start_time
    print("程序运行时间为：", runtime, "秒")
