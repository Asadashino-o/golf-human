import cv2
import torch
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor, ScoredBoundingBoxExtractor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import os
import numpy as np


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


def initialize_det_segmentation():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = DefaultPredictor(cfg)
    return model


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
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

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
    selected_indices = [i for i, score in enumerate(scores) if score > 0.9]
    # 根据筛选后的索引从 densepose_results 和 boxes_xywh 中提取对应的值
    filtered_densepose_results = [densepose_results[i] for i in selected_indices]
    filtered_boxes_xywh = [boxes_xywh[i] for i in selected_indices]

    # 找出 box[2] 和 box[3] 最大的实例：面积最大
    if filtered_densepose_results and filtered_boxes_xywh:
        max_area_box = max(zip(filtered_densepose_results, filtered_boxes_xywh), key=lambda x: x[1][2] * x[1][3])
        return max_area_box
    else:
        # 处理空序列的情况，例如：
        return None


def inference2(predictor, frame, ball_position):
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

    if filtered_densepose_results and filtered_boxes_xywh:
        # 找出 bbox 右下角离球位置最近的实例
        closest_box = min(zip(filtered_densepose_results, filtered_boxes_xywh), key=lambda x: ((x[1][0] + x[1][2]) - ball_position[0])**2 + ((x[1][1] + x[1][3]) - ball_position[1])**2)
        return closest_box
    else:
        # 处理空序列的情况，例如：
        return None


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


def get_densepose_info(instance):
    """
    这个函数是为了获取densepose得到的最大实例的具体iuv和bbox信息
    :param instance: densepose检测得到的面积最大实例
    :return: iuv和bbox信息
    """
    if instance is None:
        return None
    result, box = instance
    iuv_array = torch.cat(
        (result.labels[None].type(torch.float32), result.uv * 255.0)
    ).type(torch.uint8)
    iuv_array = iuv_array.cpu().numpy()  # 将 CUDA tensor 转换为 NumPy 数组
    return iuv_array, box


# fine segmentation: 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand,
# 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
# 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
# 20, 22 = Lower Arm Right, 23, 24 = Head
def get_head_densepose(iuv_arr, bbox_xywh):
    """
        这个函数是为了返回被判定为头部的最上边像素的绝对坐标
        :param iuv_arr: densepose得到的iuv坐标，i记载了每个像素所属的种类,是一个三维的numpy数组
        :param bbox_xywh:实例的左上角xy坐标和宽w，高h。
        :return:臀头部最上边的绝对坐标，处理为纯数据.item()
    """
    matrix = iuv_arr[0]
    head_mask = np.isin(matrix, [23, 24])
    head_indices = np.argwhere(head_mask)

    if head_indices.size == 0:
        return None

    top_head_index = head_indices[np.argmin(head_indices[:, 0])]
    x, y, w, h = bbox_xywh
    head_x = x + top_head_index[1]
    head_y = y + top_head_index[0]

    if head_x <= x + w and head_y <= y + h:
        return head_x.item(), head_y.item()
    else:
        return None


def get_buttocks_densepose(iuv_arr, bbox_xywh):
    """
    这个函数是为了返回被判定为臀的最左边像素的绝对坐标
    :param iuv_arr: densepose得到的iuv坐标，i记载了每个像素所属的种类,是一个三维的numpy数组
    :param bbox_xywh:实例的左上角xy坐标和宽w，高h。
    :return:臀部最左边的绝对坐标，处理为纯数据.item()
    """
    matrix = iuv_arr[0]
    buttocks_mask = np.isin(matrix, [8, 10, 7, 9])
    buttocks_indices = np.argwhere(buttocks_mask)

    if buttocks_indices.size == 0:
        return None

    left_buttocks_index = buttocks_indices[np.argmin(buttocks_indices[:, 1])]
    x, y, w, h = bbox_xywh
    buttocks_x = x + left_buttocks_index[1]
    buttocks_y = y + left_buttocks_index[0]

    if buttocks_x <= x + w and buttocks_y <= y + h:
        return buttocks_x.item(), buttocks_y.item()
    else:
        return None


def get_headtop(instance):
    """
    返回被判定为头部的最上边像素的绝对坐标
    :param instance: detectron2检测得到的最近的头部实例
    :return: 头部最上边的绝对坐标 (x, y), 处理为纯数据.item()
    """
    if instance is None:
        return None

    # 获取头部的 mask
    pred_masks = instance['pred_masks'][0].cpu().numpy()

    # 获取 mask 中为 True 的像素的坐标
    mask_indices = np.argwhere(pred_masks)

    # 获取最上边的像素坐标
    top_pixel = mask_indices[mask_indices[:, 0].argmin()]  # 找到最小的 y 坐标对应的索引

    # 返回 x 和 y 坐标
    return int(top_pixel[1]), int(top_pixel[0])


def save_to_txt(x_list, y_list, bbox_list, filename):
    with open(filename, 'w') as file:
        for frame, (x_frame, y_frame, bbox_frame) in enumerate(zip(x_list, y_list, bbox_list)):
            for x, y, bbox in zip(x_frame, y_frame, bbox_frame):
                file.write(f"X: {x}, Y: {y}, Bbox: {bbox}\n")
            file.write("\n")


def save_data(x1_coordinates, y1_coordinates, x2_coordinates, y2_coordinates, bbox_info):
    save_to_txt([x1_coordinates], [y1_coordinates], [bbox_info], "info/coordinates_and_bbox_buttock.txt")
    save_to_txt([x2_coordinates], [y2_coordinates], [bbox_info], "info/coordinates_and_bbox_head.txt")


def filter_and_select_closest_instance(instances, head_coords, confidence_threshold=0.90):
    if head_coords is None:
        return None
    head_x, head_y = head_coords
    head_x, head_y = int(head_x), int(head_y)

    pred_boxes = instances.pred_boxes.tensor  # 转换为 tensor
    scores = instances.scores
    pred_classes = instances.pred_classes
    pred_masks = instances.pred_masks

    # 1. 排除置信度小于阈值的实例
    valid_indices = scores > confidence_threshold
    pred_boxes = pred_boxes[valid_indices]
    scores = scores[valid_indices]
    pred_classes = pred_classes[valid_indices]
    pred_masks = pred_masks[valid_indices]

    # 如果没有有效的实例，返回 None
    if len(pred_boxes) == 0:
        return None

    # 2. 找到所有 head 类别（类别为0）的实例
    head_indices = (pred_classes == 0).nonzero(as_tuple=True)[0]

    if len(head_indices) == 0:
        return None

    closest_instance_info = None
    min_distance = float('inf')

    for idx in head_indices:
        mask = pred_masks[idx]

        if head_y < mask.shape[0] and head_x < mask.shape[1] and mask[head_y, head_x]:
            return {
                'pred_boxes': pred_boxes[idx].unsqueeze(0).cpu(),
                'scores': scores[idx].unsqueeze(0).cpu(),
                'pred_classes': pred_classes[idx].unsqueeze(0).cpu(),
                'pred_masks': pred_masks[idx].unsqueeze(0).cpu()
            }

        mask_coords = torch.nonzero(mask, as_tuple=False)
        if mask_coords.size(0) > 0:
            distances = torch.sqrt((mask_coords[:, 0] - head_y) ** 2 + (mask_coords[:, 1] - head_x) ** 2)
            min_dist = distances.min().item()

            if min_dist < min_distance:
                min_distance = min_dist
                closest_instance_info = {
                    'pred_boxes': pred_boxes[idx].unsqueeze(0).cpu(),
                    'scores': scores[idx].unsqueeze(0).cpu(),
                    'pred_classes': pred_classes[idx].unsqueeze(0).cpu(),
                    'pred_masks': pred_masks[idx].unsqueeze(0).cpu()
                }

    return closest_instance_info
