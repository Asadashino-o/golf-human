import argparse
import cv2
import time
import numpy as np
import torch
from densepose.vis.extractor import DensePoseResultExtractor, ScoredBoundingBoxExtractor
from head_and_buttock import draw_x_rectangle, draw_y_rectangle, get_head, get_buttocks, video_prepare, \
    initialize_detector, video_realease


def inference(predictor, frame):
    with torch.no_grad():
        outputs = predictor(frame)["instances"]
    # 提取 DensePose 结果和得分框结果
    results = DensePoseResultExtractor()(outputs)
    scored_bboxes = ScoredBoundingBoxExtractor()(outputs)
    if results is None:
        return None
    # 获取人体密度信息，边界框和置信度分数
    densepose_results, boxes_xywh = results
    _, scores = scored_bboxes
    # 先对 scores 大于 95 的索引进行筛选
    selected_indices = [i for i, score in enumerate(scores) if score > 0.95]
    # 根据筛选后的索引从 densepose_results 和 boxes_xywh 中提取对应的值
    filtered_densepose_results = [densepose_results[i] for i in selected_indices]
    filtered_boxes_xywh = [boxes_xywh[i] for i in selected_indices]

    # 计算每个实例的面积并将其与对应的结果和边界框一起存储
    area_and_boxes = [(x[0], x[1], x[1][2] * x[1][3]) for x in zip(filtered_densepose_results, filtered_boxes_xywh)]

    # 按面积从大到小排序
    sorted_area_and_boxes = sorted(area_and_boxes, key=lambda x: x[2], reverse=True)

    # 选择面积前二的实例
    top_two_area_boxes = sorted_area_and_boxes[:2]

    # 按 x[1][0] 从小到大排序
    sorted_top_two_area_boxes = sorted(top_two_area_boxes, key=lambda x: x[1][0])

    # 提取前二实例的结果和边界框
    filtered_results = [(item[0], item[1]) for item in sorted_top_two_area_boxes]

    return filtered_results


def main(input_video_path, output_video_path, f_idx, interval=4):
    # Initialize Detectron2 configuration for DensePose
    predictor = initialize_detector()
    video_capture, video_writer, num_frames = video_prepare(input_video_path, output_video_path)
    # Process each frame in the video
    frame_num = 0
    begin_X = 0
    begin_Y = 0
    prev_x = 0
    prev_y = 0
    fo_up = 0
    fo_down = 0
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

    while True:
        ret, frame = video_capture.read()  # 读取一帧
        if not ret:
            break  # 视频结束，退出循环

        # Track the region to color
        colored_region = np.zeros_like(frame)

        # 处理截止的后续帧，或者隔帧抽取操作
        if frame_num >= f_idx:
            cv2.line(frame, (int(begin_X - 100), int(begin_Y)), (int(begin_X + 100), int(begin_Y)), (0, 255, 255),
                     thickness=2)
            cv2.line(frame, (int(begin_X1), int(begin_Y1 - 100)), (int(begin_X1), int(begin_Y1 + 100)), (0, 255, 255),
                     thickness=2)
            cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)), (0, 255, 255),
                     thickness=2)
            draw_y_rectangle(int(begin_X), int(fo_up), int(begin_Y), colored_region, color=0)  # 画上边的距离
            draw_y_rectangle(int(begin_X), int(begin_Y), int(fo_down), colored_region, color=1)  # 画下边的距离
            draw_x_rectangle(int(begin_Y1), int(begin_X1), int(x_right), colored_region, color=0)  # 画右边的距离
            draw_x_rectangle(int(begin_Y1), int(x_left), int(begin_X1), colored_region, color=1)  # 画左边的距离
            draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画上边的距离
            draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画下边的距离

            video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
            frame_num += 1
            print(f"Ignore frame {frame_num} / {num_frames}")
            continue

        if (frame_num % interval != 0) and (frame_num != f_idx - 1):
            if prev_x1 > begin_X1:
                cv2.line(frame, (int(prev_x1), int(begin_Y1 - 60)), (int(prev_x1), int(begin_Y1 + 60)), (0, 0, 255),
                         thickness=2)  # 在 y 坐标处画一条红色的水平线
            else:
                cv2.line(frame, (int(prev_x1), int(begin_Y1 - 60)), (int(prev_x1), int(begin_Y1 + 60)), (0, 255, 0),
                         thickness=2)  # 在 y 坐标处画一条绿色的水平线
            cv2.line(frame, (int(begin_X1), int(begin_Y1 - 100)), (int(begin_X1), int(begin_Y1 + 100)), (0, 255, 255),
                     thickness=2)
            draw_x_rectangle(int(begin_Y1), int(begin_X1), int(x_right), colored_region, color=0)  # 画右边的距离
            draw_x_rectangle(int(begin_Y1), int(x_left), int(begin_X1), colored_region, color=1)  # 画左边的距离

            if prev_y2 < begin_Y2:
                cv2.line(frame, (int(begin_X2 - 60), int(prev_y2)), (int(begin_X2 + 60), int(prev_y2)), (0, 0, 255),
                         thickness=2)  # 在 x 坐标处画一条红色的水平线
            else:
                cv2.line(frame, (int(begin_X2 - 60), int(prev_y2)), (int(begin_X2 + 60), int(prev_y2)), (0, 255, 0),
                         thickness=2)  # 在 x 坐标处画一条绿色的水平线
            cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)), (0, 255, 255),
                     thickness=2)
            draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画上面的距离
            draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画下面的距离

            if prev_y < begin_Y:
                cv2.line(frame, (int(begin_X - 60), int(prev_y)), (int(begin_X + 60), int(prev_y)), (0, 0, 255),
                         thickness=2)  # 在 x 坐标处画一条红色的水平线
            else:
                cv2.line(frame, (int(begin_X - 60), int(prev_y)), (int(begin_X + 60), int(prev_y)), (0, 255, 0),
                         thickness=2)  # 在 x 坐标处画一条绿色的水平线
            cv2.line(frame, (int(begin_X - 100), int(begin_Y)), (int(begin_X + 100), int(begin_Y)), (0, 255, 255),
                     thickness=2)
            draw_y_rectangle(int(begin_X), int(fo_up), int(begin_Y), colored_region, color=0)  # 画上面的距离
            draw_y_rectangle(int(begin_X), int(begin_Y), int(fo_down), colored_region, color=1)  # 画下面的距离

            video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
            frame_num += 1
            print(f"Ignore frame {frame_num} / {num_frames}")
            continue

        filtered_results = inference(predictor, frame)
        if len(filtered_results) != 2:
            video_writer.write(frame)
            frame_num += 1
            print(f"This frame {frame_num} / {num_frames} does not have exactly 2 detected instances")
            continue
        # 分别提取这两个实例
        (result1, box1), (result2, box2) = filtered_results
        iuv_array1 = torch.cat(
            (result1.labels[None].type(torch.float32), result1.uv * 255.0)
        ).type(torch.uint8)
        iuv_array2 = torch.cat(
            (result2.labels[None].type(torch.float32), result2.uv * 255.0)
        ).type(torch.uint8)
        iuv_array1 = iuv_array1.cpu().numpy()  # 将 CUDA tensor 转换为 NumPy 数组
        iuv_array2 = iuv_array2.cpu().numpy()
        head_coords1 = get_head(iuv_array1, box1)
        head_coords2 = get_head(iuv_array2, box2)
        buttocks_coords2 = get_buttocks(iuv_array2, box2)
        if buttocks_coords2 is None:
            x1 = prev_x1
            y1 = prev_y1
        else:
            x1, y1 = buttocks_coords2
        if head_coords2 is None:
            x2 = prev_x2
            y2 = prev_y2
        else:
            x2, y2 = head_coords2
        if head_coords1 is None:
            x = prev_x
            y = prev_y
        else:
            x, y = head_coords1

        if frame_num == 0:
            prev_x1 = x1
            prev_y1 = y1
            prev_x2 = x2
            prev_y2 = y2
            prev_x = x
            prev_y = y
            begin_X = x
            begin_Y = y
            begin_X1 = x1
            begin_Y1 = y1
            begin_X2 = x2
            begin_Y2 = y2
            fo_up = begin_Y
            fo_down = begin_Y
            x_left = begin_X1
            x_right = begin_X1
            y_up = begin_Y2
            y_down = begin_Y2
            x, y, width, height = box2

        if abs(x1 - prev_x1) > 20 or abs(y1 - prev_y1) > 20:
            x1 = prev_x1
            y1 = prev_y1
        if abs(x2 - prev_x2) > 20 or abs(y2 - prev_y2) > 20:
            x2 = prev_x2
            y2 = prev_y2
        if abs(x - prev_x) > 20 or abs(y - prev_y) > 20:
            x = prev_x
            y = prev_y
        prev_x1 = x1
        prev_y1 = y1
        prev_x2 = x2
        prev_y2 = y2
        prev_x = x
        prev_y = y
        y_up = min(y_up, y2)
        y_down = max(y_down, y2)
        x_left = min(x_left, x1)
        x_right = max(x_right, x1)
        fo_up = min(fo_up, y)
        fo_down = max(fo_down, y)

        # 在图像上绘制线条
        if x1 > begin_X1:
            cv2.line(frame, (int(x1), int(begin_Y1 - 60)), (int(x1), int(begin_Y1 + 60)), (0, 0, 255),
                     thickness=2)  # 在 y 坐标处画一条红色的水平线
        else:
            cv2.line(frame, (int(x1), int(begin_Y1 - 60)), (int(x1), int(begin_Y1 + 60)), (0, 255, 0),
                     thickness=2)  # 在 y 坐标处画一条绿色的水平线
        cv2.line(frame, (int(begin_X1), int(begin_Y1 - 100)), (int(begin_X1), int(begin_Y1 + 100)), (0, 255, 255),
                 thickness=2)
        draw_x_rectangle(int(begin_Y1), int(begin_X1), int(x_right), colored_region, color=0)  # 画右边的距离
        draw_x_rectangle(int(begin_Y1), int(x_left), int(begin_X1), colored_region, color=1)  # 画左边的距离

        if y2 < begin_Y2:
            cv2.line(frame, (int(begin_X2 - 60), int(y2)), (int(begin_X2 + 60), int(y2)), (0, 0, 255),
                     thickness=2)  # 在 x 坐标处画一条红色的水平线
        else:
            cv2.line(frame, (int(begin_X2 - 60), int(y2)), (int(begin_X2 + 60), int(y2)), (0, 255, 0),
                     thickness=2)  # 在 x 坐标处画一条绿色的水平线
        cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)), (0, 255, 255),
                 thickness=2)
        draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画上面的距离
        draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画下面的距离

        if y < begin_Y:
            cv2.line(frame, (int(begin_X - 60), int(y)), (int(begin_X + 60), int(y)), (0, 0, 255),
                     thickness=2)  # 在 x 坐标处画一条红色的水平线
        else:
            cv2.line(frame, (int(begin_X - 60), int(y)), (int(begin_X + 60), int(y)), (0, 255, 0),
                     thickness=2)  # 在 x 坐标处画一条绿色的水平线
        cv2.line(frame, (int(begin_X - 100), int(begin_Y)), (int(begin_X + 100), int(begin_Y)), (0, 255, 255),
                 thickness=2)
        draw_y_rectangle(int(begin_X), int(fo_up), int(begin_Y), colored_region, color=0)  # 画上面的距离
        draw_y_rectangle(int(begin_X), int(begin_Y), int(fo_down), colored_region, color=1)  # 画下面的距离

        video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
        frame_num += 1
        print(f"Processed frame {frame_num} / {num_frames}")

    # Release resources
    message = video_realease(video_capture, video_writer, begin_X1, begin_Y2, x_left, x_right, y_up, y_down, width,
                             height)
    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    parser.add_argument("-f", "--f_idx", type=int)
    parser.add_argument("-n", "--interval", type=int)
    args = parser.parse_args()

    # 记录开始时间
    start_time = time.time()

    # 主函数
    main(args.input_video_path, args.output_video_path, args.f_idx, args.interval)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    runtime = end_time - start_time
    print("程序运行时间为：", runtime, "秒")
