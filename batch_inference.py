import argparse
import cv2
import time
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from head_and_buttock import video_prepare, initialize_detector, video_realease, inference
from head_and_buttock import save_data, draw_y_rectangle, draw_x_rectangle, get_head, get_buttocks


def main(input_video_path, output_video_path, f_idx, interval, num_frames_per_batch):
    # Initialize Detectron2 configuration for DensePose
    predictor = initialize_detector()
    video_capture, video_writer, num_frames = video_prepare(input_video_path, output_video_path)

    # Initialize lists to store x and y coordinates
    x1_coordinates = []
    y1_coordinates = []
    x2_coordinates = []
    y2_coordinates = []
    bbox_info = []

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

    while video_capture.isOpened():
        # 读取一帧
        frames = []
        for _ in range(num_frames_per_batch * interval):
            if frame_num >= f_idx:
                break
            ret, frame = video_capture.read()
            frame_num += 1
            if not ret:
                break
            frames.append(frame)
        # 如果没有帧了，则退出循环
        if not frames:
            break
        filtered_results = inference(predictor, frames, interval, num_frames_per_batch)
        idx = 0
        for i, frame in enumerate(frames):
            colored_region = np.zeros_like(frames[i])
            if i % interval == 0:
                filtered_result = filtered_results[idx]
                idx += 1
                if filtered_results is None:
                    video_writer.write(frames[i])
                    print(f"This frame {i} / {frame_num} / {num_frames} can not be detected")
                    continue
                for result, box in filtered_result:
                    iuv_array = torch.cat(
                        (result.labels[None].type(torch.float32), result.uv * 255.0)
                    ).type(torch.uint8)
                    iuv_array = iuv_array.cpu().numpy()  # 将 CUDA tensor 转换为 NumPy 数组
                    buttocks_coords = get_buttocks(iuv_array, box)
                    head_coords = get_head(iuv_array, box)
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

                    if frame_num == num_frames_per_batch * interval and i == 0:
                        prev_x1 = x1
                        prev_y1 = y1
                        prev_x2 = x2
                        prev_y2 = y2
                        begin_X1 = x1
                        begin_Y1 = y1
                        begin_X2 = x2
                        begin_Y2 = y2
                        x_left = begin_X1
                        x_right = begin_X1
                        y_up = begin_Y2
                        y_down = begin_Y2
                        x, y, w, h = box
                        width = w
                        height = h
                        # area = area_calculate(iuv_array, w, h)

                    if abs(x1 - prev_x1) > 20 or abs(y1 - prev_y1) > 20:
                        x1 = prev_x1
                        y1 = prev_y1
                    if abs(x2 - prev_x2) > 20 or abs(y2 - prev_y2) > 20:
                        x2 = prev_x2
                        y2 = prev_y2

                    x1_coordinates.append(x1)
                    y1_coordinates.append(y1)
                    x2_coordinates.append(x2)
                    y2_coordinates.append(y2)
                    bbox_info.append(box.tolist())
                    prev_x1 = x1
                    prev_y1 = y1
                    prev_x2 = x2
                    prev_y2 = y2

                    x_left = min(x_left, x1)
                    x_right = max(x_right, x1)
                    y_up = min(y_up, y2)
                    y_down = max(y_down, y2)
                    # 在图像上绘制线条
                    if x1 > begin_X1:
                        cv2.line(frame, (int(x1), int(y1 - 60)), (int(x1), int(y1 + 60)), (0, 0, 255),
                                 thickness=2)  # 在 y 坐标处画一条红色的水平线
                    else:
                        cv2.line(frame, (int(x1), int(y1 - 60)), (int(x1), int(y1 + 60)), (0, 255, 0),
                                 thickness=2)  # 在 y 坐标处画一条绿色的水平线
                    cv2.line(frame, (int(begin_X1), int(begin_Y1 - 100)), (int(begin_X1), int(begin_Y1 + 100)),
                             (0, 255, 255),
                             thickness=2)
                    draw_x_rectangle(int(begin_Y1), int(begin_X1), int(x_right), colored_region, color=0)  # 画右边的距离
                    draw_x_rectangle(int(begin_Y1), int(x_left), int(begin_X1), colored_region, color=1)  # 画左边的距离

                    if y2 < begin_Y2:
                        cv2.line(frame, (int(x2 - 60), int(y2)), (int(x2 + 60), int(y2)), (0, 0, 255),
                                 thickness=2)  # 在 x 坐标处画一条红色的水平线
                    else:
                        cv2.line(frame, (int(x2 - 60), int(y2)), (int(x2 + 60), int(y2)), (0, 255, 0),
                                 thickness=2)  # 在 x 坐标处画一条绿色的水平线
                    cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)),
                             (0, 255, 255),
                             thickness=2)
                    draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画上面的距离
                    draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画下面的距离

                video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))

                print(f"Processed frame {i} / {frame_num} / {num_frames}")
            else:
                cv2.line(frame, (int(begin_X1), int(begin_Y1 - 100)), (int(begin_X1), int(begin_Y1 + 100)), (0, 255, 255),
                         thickness=2)
                cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)), (0, 255, 255),
                         thickness=2)
                draw_x_rectangle(int(begin_Y1), int(begin_X1), int(x_right), colored_region, color=0)  # 画右边的距离
                draw_x_rectangle(int(begin_Y1), int(x_left), int(begin_X1), colored_region, color=1)  # 画左边的距离
                draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画右边的距离
                draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画左边的距离

                video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
                print(f"Ignore frame {i} / {frame_num} / {num_frames}")
        # 处理后续帧
        while frame_num >= f_idx:
            ret, frame = video_capture.read()
            if not ret:
                break
            colored_region = np.zeros_like(frame)
            cv2.line(frame, (int(begin_X1), int(begin_Y1 - 100)), (int(begin_X1), int(begin_Y1 + 100)), (0, 255, 255),
                     thickness=2)
            cv2.line(frame, (int(begin_X2 - 100), int(begin_Y2)), (int(begin_X2 + 100), int(begin_Y2)), (0, 255, 255),
                     thickness=2)
            draw_x_rectangle(int(begin_Y1), int(begin_X1), int(x_right), colored_region, color=0)  # 画右边的距离
            draw_x_rectangle(int(begin_Y1), int(x_left), int(begin_X1), colored_region, color=1)  # 画左边的距离
            draw_y_rectangle(int(begin_X2), int(y_up), int(begin_Y2), colored_region, color=0)  # 画右边的距离
            draw_y_rectangle(int(begin_X2), int(begin_Y2), int(y_down), colored_region, color=1)  # 画左边的距离

            video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
            frame_num += 1
            print(f"Processed frame {frame_num} / {num_frames}")

    # Release resources
    video_realease(video_capture, video_writer, begin_X1, begin_Y2, x_left, x_right, y_up, y_down, width, height)
    save_data(x1_coordinates, y1_coordinates, x2_coordinates, y2_coordinates, bbox_info)


class VideoBatchDefaultPredictor(DefaultPredictor):
    """
    A batch version of the DefaultPredictor class to process a batch of video frames.

    Args:
        cfg (CfgNode): the config. Options: MODEL.DEVICE, MODEL.WEIGHTS, DATASETS.TEST,
        INPUT.MIN_SIZE_TEST, INPUT.MAX_SIZE_TEST, INPUT.FORMAT.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from cfg.DATASETS.TEST.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_frames, num_frames=1):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_frames = [frame[:, :, ::-1] for frame in original_frames]

            augmented_frames = []
            for frame in original_frames:
                height, width = frame.shape[:2]
                frame = self.aug.get_transform(frame).apply_image(frame)
                frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
                augmented_frames.append({"image": frame, "height": height, "width": width})

            predictions = []
            for i in range(0, len(augmented_frames), num_frames):
                batch_frames = augmented_frames[i:i + num_frames]
                batch_inputs = [{"image": frame["image"], "height": frame["height"], "width": frame["width"]} for frame
                                in batch_frames]
                batch_predictions = self.model(batch_inputs)
                predictions.extend(batch_predictions)

            return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    parser.add_argument("-f", "--f_idx", type=int)
    parser.add_argument("-d", "--interval", type=int)
    parser.add_argument("-b", "--batch_size", type=int)
    args = parser.parse_args()

    # 记录开始时间
    start_time = time.time()

    # 主函数
    main(args.input_video_path, args.output_video_path, args.f_idx, args.interval, args.batch_size)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    runtime = end_time - start_time
    print("程序运行时间为：", runtime, "秒")
