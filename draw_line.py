import argparse
import cv2
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor, ScoredBoundingBoxExtractor

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def main(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4", f_idx=604, num_frames_per_batch=4):
    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    batch_predictor = VideoBatchDefaultPredictor(cfg)

    # Open the input video
    video_capture = cv2.VideoCapture(input_video_path)

    # 定义每次处理几帧
    # num_frames_per_batch = 4

    # Get the video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize lists to store x and y coordinates
    x1_coordinates = []
    y1_coordinates = []
    x2_coordinates = []
    y2_coordinates = []
    bbox_info = []

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
        for _ in range(num_frames_per_batch):
            ret, frame = video_capture.read()
            frame_num += 1
            if not ret:
                break
            frames.append(frame)

        # 如果没有帧了，则退出循环
        if not frames:
            break
        # 处理当前帧
        with torch.no_grad():
            outputs = batch_predictor(frames, num_frames=num_frames_per_batch)
        for i, pred in enumerate(outputs):
            output = pred["instances"]
            # 提取 DensePose 结果和得分框结果
            results = DensePoseResultExtractor()(output)
            scored_bboxes = ScoredBoundingBoxExtractor()(output)
            # Track the region to color
            colored_region = np.zeros_like(frames[i])
            if results is None:
                video_writer.write(frames[i])
                print(f"Processed frame {frame_num} / {num_frames}")
                continue
            densepose_results, boxes_xywh = results
            _, scores = scored_bboxes

            # 先对 scores 大于 95 的索引进行筛选
            selected_indices = [i for i, score in enumerate(scores) if score > 0.95]
            # 根据筛选后的索引从 densepose_results 和 boxes_xywh 中提取对应的值
            filtered_densepose_results = [densepose_results[i] for i in selected_indices]
            filtered_boxes_xywh = [boxes_xywh[i] for i in selected_indices]

            # 找出 box[2] 和 box[3] 最大的实例：面积最大
            max_area_box = max(zip(filtered_densepose_results, filtered_boxes_xywh), key=lambda x: x[1][2] * x[1][3])
            filtered_results = [max_area_box]

            if filtered_results is None:
                video_writer.write(frames[i])
                print(f"Processed frame {frame_num} / {num_frames}")
                continue
            for result, box in filtered_results:
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

                x1_coordinates.append(x1)
                y1_coordinates.append(y1)
                x2_coordinates.append(x2)
                y2_coordinates.append(y2)
                bbox_info.append(box.tolist())
                prev_x1 = x1
                prev_y1 = y1
                prev_x2 = x2
                prev_y2 = y2

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                if frame_num == num_frames_per_batch and i == 0:
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
                x_left = min(x_left, x1)
                x_right = max(x_right, x1)
                y_up = min(y_up, y2)
                y_down = max(y_down, y2)

                # 在图像上绘制线条
                cv2.line(frames[i], (x1, y1 - 60), (x1, y1 + 60), (0, 0, 255), thickness=2)  # 在 y 坐标处画一条绿色的水平线
                cv2.line(frames[i], (begin_X1, begin_Y1 - 100), (begin_X1, begin_Y1 + 100), (0, 255, 0), thickness=2)
                draw_x_rectangle(begin_Y1, begin_X1, x_right, colored_region, color=0)  # 画右边的距离
                draw_x_rectangle(begin_Y1, x_left, begin_X1, colored_region, color=1)  # 画左边的距离

                cv2.line(frames[i], (x2 - 60, y2), (x2 + 60, y2), (0, 0, 255), thickness=2)  # 在 x 坐标处画一条红色的水平线
                cv2.line(frames[i], (begin_X2 - 100, begin_Y2), (begin_X2 + 100, begin_Y2), (0, 255, 0), thickness=2)
                draw_y_rectangle(begin_X2, y_up, begin_Y2, colored_region, color=0)  # 画上面的距离
                draw_y_rectangle(begin_X2, begin_Y2, y_down, colored_region, color=1)  # 画下面的距离

                video_writer.write(cv2.addWeighted(frames[i], 0.8, colored_region, 0.2, 0))
                print(f"Processed frame {frame_num} / {num_frames}")
        # 处理后续帧
        while frame_num >= f_idx:
            ret, frame = video_capture.read()
            if not ret:
                break
            colored_region = np.zeros_like(frame)
            cv2.line(frame, (begin_X1, begin_Y1 - 100), (begin_X1, begin_Y1 + 100), (0, 255, 0), thickness=2)
            cv2.line(frame, (begin_X2 - 100, begin_Y2), (begin_X2 + 100, begin_Y2), (0, 255, 0), thickness=2)
            draw_x_rectangle(begin_Y1, begin_X1, x_right, colored_region, color=0)  # 画右边的距离
            draw_x_rectangle(begin_Y1, x_left, begin_X1, colored_region, color=1)  # 画左边的距离
            draw_y_rectangle(begin_X2, y_up, begin_Y2, colored_region, color=0)  # 画右边的距离
            draw_y_rectangle(begin_X2, begin_Y2, y_down, colored_region, color=1)  # 画左边的距离

            video_writer.write(cv2.addWeighted(frame, 0.8, colored_region, 0.2, 0))
            frame_num += 1
            print(f"Processed frame {frame_num} / {num_frames}")
            continue

    # Release resources
    video_capture.release()
    video_writer.release()
    dis_left = (begin_X1 - x_left) / width
    dis_right = (x_right - begin_X1) / width
    dis_up = (begin_Y2 - y_up) / height
    dis_down = (y_down - begin_Y2) / height
    print("臀部向左相对移动距离：{}".format(dis_left))
    print("臀部向右相对移动距离：{}".format(dis_right))
    print("头部向上相对移动距离：{}".format(dis_up))
    print("头部向下相对移动距离：{}".format(dis_down))
    x1_smooth, y1_smooth = smooth_coordinates(x1_coordinates, y1_coordinates)
    x2_smooth, y2_smooth = smooth_coordinates(x2_coordinates, y2_coordinates)
    save_to_txt([x1_coordinates], [y1_coordinates], [bbox_info], "0415/coordinates_and_bbox_buttock.txt")
    save_to_txt([x1_smooth], [y1_smooth], [bbox_info], "0415/smooth_and_bbox_buttock.txt")
    save_to_txt([x2_coordinates], [y2_coordinates], [bbox_info], "0415/coordinates_and_bbox_head.txt")
    save_to_txt([x2_smooth], [y2_smooth], [bbox_info], "0415/smooth_and_bbox_head.txt")
    # plot_xy(x_coordinates, y_coordinates, x_smooth, y_smooth)


def draw_x_rectangle(begin_Y, x_left, x_right, colored_region, color=0):
    # Draw left colored region
    if color == 0:
        cv2.rectangle(colored_region, (x_left, begin_Y - 60), (x_right, begin_Y + 60), (0, 0, 255),
                      thickness=cv2.FILLED)
    elif color == 1:
        cv2.rectangle(colored_region, (x_left, begin_Y - 60), (x_right, begin_Y + 60), (0, 255, 0),
                      thickness=cv2.FILLED)


def draw_y_rectangle(begin_X, y_up, y_down, colored_region, color=0):
    # Draw top colored region
    if color == 0:
        cv2.rectangle(colored_region, (begin_X - 60, y_up), (begin_X + 60, y_down), (255, 0, 0),
                      thickness=cv2.FILLED)
    elif color == 1:
        cv2.rectangle(colored_region, (begin_X - 60, y_up), (begin_X + 60, y_down), (255, 255, 0),
                      thickness=cv2.FILLED)


def plot_xy(x_coordinates, y_coordinates, x_smooth, y_smooth):
    # Plot both original and smoothed coordinates
    plt.figure(figsize=(10, 5))

    # Plot original x coordinates
    plt.subplot(1, 2, 1)
    plt.plot(range(len(x_coordinates)), x_coordinates, label='Original')
    plt.plot(range(len(x_smooth)), x_smooth, label='Smoothed')  # 添加平滑后的曲线
    plt.xlabel('Frame Number')
    plt.ylabel('X Coordinate')
    plt.title('X Coordinate vs Frame Number')
    plt.legend()

    # Plot original y coordinates
    plt.subplot(1, 2, 2)
    plt.plot(range(len(y_coordinates)), y_coordinates, label='Original')
    plt.plot(range(len(y_smooth)), y_smooth, label='Smoothed')  # 添加平滑后的曲线
    plt.xlabel('Frame Number')
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinate vs Frame Number')
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig('coordinates_plot.png')


# fine segmentation: 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand,
# 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
# 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
# 20, 22 = Lower Arm Right, 23, 24 = Head
def get_head(iuv_arr, bbox_xywh):
    """
        这个函数是为了返回被判定为头部的最上边像素的绝对坐标
        :param iuv_arr: iuv坐标，i记载了每个像素所属的种类,是一个三维的numpy数组
        :param bbox_xywh:实例的左上角xy坐标和宽w，高h。
        :return:臀头部最上边的绝对坐标,tensor格式，处理为纯数据.item()
    """
    matrix = iuv_arr[0]
    index_x = 0
    index_y = 10000
    for i in range(len(matrix[0])):  # 使用range函数以获取正确的索引
        for j in range(len(matrix)):  # 使用range函数以获取正确的索引
            if int(matrix[j][i]) in {23, 24}:
                if index_y > j:  # 更新最上边像素的索引
                    index_y = j
                    index_x = i
    x, y, w, h = bbox_xywh
    if index_x <= w and index_y <= h:
        return (x + index_x).item(), (y + index_y).item()
    else:
        return None


def get_buttocks(iuv_arr, bbox_xywh):
    """
    这个函数是为了返回被判定为臀的最左边像素的绝对坐标
    :param iuv_arr: iuv坐标，i记载了每个像素所属的种类,是一个三维的numpy数组
    :param bbox_xywh:实例的左上角xy坐标和宽w，高h。
    :return:臀部最左边的绝对坐标,tensor格式，处理为纯数据.item()
    """
    matrix = iuv_arr[0]
    index_x = 10000
    index_y = 0
    for i in range(len(matrix)):  # 使用range函数以获取正确的索引
        for j in range(len(matrix[0])):  # 使用range函数以获取正确的索引
            if int(matrix[i][j]) in {8, 10, 7, 9}:
                if index_x > j:  # 更新最左边像素的索引
                    index_x = j
                    index_y = i
    x, y, w, h = bbox_xywh
    if index_x <= w and index_y <= h:
        return (x + index_x).item(), (y + index_y).item()
    else:
        return None


def smooth_coordinates(x_coordinates, y_coordinates, window_size=5, threshold=0.2):
    smoothed_x = []
    smoothed_y = []

    for i in range(len(x_coordinates)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(x_coordinates), i + window_size // 2 + 1)
        window_x = x_coordinates[start_index:end_index]
        window_y = y_coordinates[start_index:end_index]

        # Compute moving average
        avg_x = sum(window_x) / len(window_x)
        avg_y = sum(window_y) / len(window_y)

        # Check for outliers
        if abs(x_coordinates[i] - avg_x) > threshold * avg_x or abs(y_coordinates[i] - avg_y) > threshold * avg_y:
            # If the point is an outlier, use the previous smoothed value
            smoothed_x.append(smoothed_x[-1] if smoothed_x else x_coordinates[i])
            smoothed_y.append(smoothed_y[-1] if smoothed_y else y_coordinates[i])
        else:
            smoothed_x.append(avg_x)
            smoothed_y.append(avg_y)

    return smoothed_x, smoothed_y


def save_to_txt(x_list, y_list, bbox_list, filename):
    with open(filename, 'w') as file:
        for frame, (x_frame, y_frame, bbox_frame) in enumerate(zip(x_list, y_list, bbox_list)):
            for x, y, bbox in zip(x_frame, y_frame, bbox_frame):
                file.write(f"X: {x}, Y: {y}, Bbox: {bbox}\n")
            file.write("\n")


def area_calculate(iuv_arr, box_w, box_h):
    num = 0
    matrix = iuv_arr[0]
    for i in range(len(matrix)):  # 使用range函数以获取正确的索引
        for j in range(len(matrix[0])):  # 使用range函数以获取正确的索引
            if int(matrix[i][j]) == 0:
                num += 1
    area = box_w * box_h - num
    return area


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
    args = parser.parse_args()
    f_idx = int(input("请输入视频检测截止帧："))
    num_frames_per_batch = int(input("请输入一批次的图片数量："))

    # 记录开始时间
    start_time = time.time()

    main(args.input_video_path, args.output_video_path, f_idx, num_frames_per_batch)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    runtime = end_time - start_time
    print("程序运行时间为：", runtime, "秒")
