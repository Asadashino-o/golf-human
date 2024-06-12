import cv2
import torch
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor, ScoredBoundingBoxExtractor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
from matplotlib import pyplot as plt


def initialize_detector():
    cfg = get_cfg()
    add_densepose_config(cfg)
    if os.path.exists("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"):
        cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    else:
        cfg.merge_from_file("densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor


def video_prepare(input_video_path, output_video_path):
    # Open the input video
    video_capture = cv2.VideoCapture(input_video_path)

    # Get the video properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 视频编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    return video_capture, video_writer, num_frames


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

    # 找出 box[2] 和 box[3] 最大的实例：面积最大
    max_area_box = max(zip(filtered_densepose_results, filtered_boxes_xywh), key=lambda x: x[1][2] * x[1][3])
    filtered_results = [max_area_box]
    return filtered_results


def video_realease(video_capture, video_writer, begin_X1, begin_Y2, x_left, x_right, y_up, y_down, width, height,
                   message=""):
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
    message += "臀部向左相对移动距离：{}\n".format(dis_left)
    message += "臀部向右相对移动距离：{}\n".format(dis_right)
    message += "头部向上相对移动距离：{}\n".format(dis_up)
    message += "头部向下相对移动距离：{}\n".format(dis_down)
    return message


def save_data(x1_coordinates, y1_coordinates, x2_coordinates, y2_coordinates, bbox_info):
    x1_smooth, y1_smooth = smooth_coordinates(x1_coordinates, y1_coordinates)
    x2_smooth, y2_smooth = smooth_coordinates(x2_coordinates, y2_coordinates)
    save_to_txt([x1_coordinates], [y1_coordinates], [bbox_info], "info/coordinates_and_bbox_buttock.txt")
    save_to_txt([x1_smooth], [y1_smooth], [bbox_info], "info/smooth_and_bbox_buttock.txt")
    save_to_txt([x2_coordinates], [y2_coordinates], [bbox_info], "info/coordinates_and_bbox_head.txt")
    save_to_txt([x2_smooth], [y2_smooth], [bbox_info], "info/smooth_and_bbox_head.txt")


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
        cv2.rectangle(colored_region, (begin_X - 60, y_up), (begin_X + 60, y_down), (0, 0, 255),
                      thickness=cv2.FILLED)
    elif color == 1:
        cv2.rectangle(colored_region, (begin_X - 60, y_up), (begin_X + 60, y_down), (0, 255, 0),
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
