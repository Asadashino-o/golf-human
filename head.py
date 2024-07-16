import argparse
import cv2
import time
import numpy as np
from Tools import inference2, draw_y_rectangle, get_headtop, video_prepare, initialize_detector, get_densepose_info, \
    get_head_densepose, initialize_det_segmentation, filter_and_select_closest_instance


def head(input_video_path, output_video_path, f_idx, interval, ball_position, message=""):
    predictor = initialize_detector()
    model = initialize_det_segmentation()
    video_capture, video_writer, num_frames = video_prepare(input_video_path, output_video_path)

    # Process each frame in the video
    frame_num = 0
    begin_X2 = 0
    begin_Y2 = 0
    y_up = 0
    y_down = 0
    prev_x2 = 0
    prev_y2 = 0
    height = 0

    while True:
        ret, frame = video_capture.read()  # 读取一帧
        if not ret:
            break  # 视频结束，退出循环

        # Track the region to color
        colored_region = np.zeros_like(frame)

        # 处理后续帧
        if frame_num >= f_idx:
            cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)), (0, 255, 255),
                     thickness=2)
            draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画右边的距离
            draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画左边的距离

            video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
            frame_num += 1
            print(f"Ignore frame {frame_num} / {num_frames}")
            continue

        if (frame_num % interval != 0) and (frame_num != f_idx - 1):
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

            video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
            frame_num += 1
            print(f"Ignore frame {frame_num} / {num_frames}")
            continue

        filtered_results = inference2(predictor, frame, ball_position)
        outputs = model(frame)["instances"]
        if filtered_results is None:
            video_writer.write(frame)
            frame_num += 1
            print(f"This frame {frame_num} / {num_frames} can not be detected")
            continue

        iuv_array, box = get_densepose_info(filtered_results)
        head_coords = get_head_densepose(iuv_array, box)
        head_instance_info = filter_and_select_closest_instance(outputs, head_coords)
        head_coords = get_headtop(head_instance_info)
        if head_coords is None:
            x2 = prev_x2
            y2 = prev_y2
        else:
            x2, y2 = head_coords

        if frame_num == 0:
            begin_X2 = x2
            begin_Y2 = y2
            prev_x2 = x2
            prev_y2 = y2
            y_up = begin_Y2
            y_down = begin_Y2
            _, _, _, height = box

        if abs(x2 - prev_x2) > 20 or abs(y2 - prev_y2) > 20:
            x2 = prev_x2
            y2 = prev_y2

        prev_x2 = x2
        prev_y2 = y2

        y_up = min(y_up, y2)
        y_down = max(y_down, y2)

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

        video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))

        frame_num += 1
        print(f"Processed frame {frame_num} / {num_frames}")

    # Release resources
    video_capture.release()
    video_writer.release()
    dis_up = (begin_Y2 - y_up) / height
    dis_down = (y_down - begin_Y2) / height
    print("头部向上相对移动距离：{}".format(dis_up))
    print("头部向下相对移动距离：{}".format(dis_down))
    message += "头部向上相对移动距离：{}\n".format(dis_up)
    message += "头部向下相对移动距离：{}\n".format(dis_down)
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
    parser.add_argument("-in", "--interval", type=int)
    args = parser.parse_args()

    # 记录开始时间
    start_time = time.time()

    head(args.input_video_path, args.output_video_path, args.f_idx, args.interval, None)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    runtime = end_time - start_time
    print("程序运行时间为：", runtime, "秒")
