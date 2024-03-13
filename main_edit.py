import argparse
import os
import cv2
import time
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def main(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4"):
    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create the output folder if it doesn't exist
    output_folder="./video_information"
    os.makedirs(output_folder, exist_ok=True)

    # Process each frame
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            print(f"frame {frame_index} ")
            t0 = time.time()
            outputs = predictor(frame)["instances"]
            print("inference time ", time.time() - t0, "s")
            # Save results to a text file
            save_results_to_txt(outputs, output_folder, frame_index)

        results = DensePoseResultExtractor()(outputs)


        # MagicAnimate uses the Viridis colormap for their training data
        cmap = cv2.COLORMAP_VIRIDIS
        # Visualizer outputs black for background, but we want the 0 value of
        # the colormap, so we initialize the array with that value
        arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
        out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
        out.write(out_frame)

        frame_index += 1

    # Release resources
    cap.release()
    out.release()

def save_results_to_txt(outputs, output_folder, frame_index):
    # 从 Instances 对象中提取边界框坐标和 pred_densepose 信息
    bboxes = outputs.pred_boxes.tensor.cpu().numpy()
    densepose_labels = outputs.pred_densepose  # 提取 pred_densepose 信息
    print(densepose_labels.coarse_segm.size())
    print(densepose_labels.fine_segm.size())

    # 创建文件名，假设使用 frame_index 作为文件名
    filename = f"{output_folder}/frame_{frame_index}.txt"

    # 打开文件并写入坐标和 pred_densepose 信息
    with open(filename, 'w') as file:
        for i, (bbox, densepose_label) in enumerate(zip(bboxes, densepose_labels)):
            x_min, y_min, x_max, y_max = bbox
            file.write(f"BBox {i+1}: {x_min},{y_min},{x_max},{y_max}\n")
            file.write(f"DensePose {i+1}:\n{densepose_label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    args = parser.parse_args()

    main(args.input_video_path, args.output_video_path)
