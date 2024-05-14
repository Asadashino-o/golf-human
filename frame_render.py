import argparse
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor
from head_and_buttock import initialize_detector


def main(input_image_path="./input_image.jpg"):
    predictor = initialize_detector()

    # Read the input image
    frame = cv2.imread(input_image_path)
    # 获取图像的高度和宽度
    height, width, _ = frame.shape

    with torch.no_grad():
        outputs = predictor(frame)["instances"]

    results = DensePoseResultExtractor()(outputs)

    densepose_results, boxes_xywh = results

    # 找出 box[2] 和 box[3] 最大的实例
    max_area_box = max(zip(densepose_results, boxes_xywh), key=lambda x: x[1][2] * x[1][3])

    # 使用找到的最大实例
    filtered_results = [max_area_box]

    for result, box in filtered_results:
        iuv_array = torch.cat(
            (result.labels[None].type(torch.float32), result.uv * 255.0)
        ).type(torch.uint8)
        iuv_array = iuv_array.cpu().numpy()  # 将 CUDA tensor 转换为 NumPy 数组
        print(iuv_array.shape)

        mask, matrix = get_mask_matrix(iuv_array)

        # 将原始图像传递给 visualize 函数进行处理
        result_image, frame = visualize(frame, mask, matrix, box, 0, 1)
        cv2.imwrite("body_data_dtl/all_image.jpg", result_image)
        result_image, frame = visualize(frame, mask, matrix, box, 0, 0)
        cv2.imwrite("body_data_dtl/full_image.jpg", result_image)
        for i in range(1, 25):
            result_frame, frame = visualize(frame, mask, matrix, box, i, 2)
            cv2.imwrite(f"body_data_dtl/{i}.jpg", result_frame)

        # 或者将结果图像保存到文件 cv2.imwrite("result_image.jpg", result_image)


def get_mask_matrix(iuv_array):
    matrix = iuv_array[0]
    segm = iuv_array[0]
    mask = np.zeros(matrix.shape, dtype=np.uint8)
    mask[segm > 0] = 1
    return mask, matrix


# fine segmentation: 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand,
# 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
# 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
# 20, 22 = Lower Arm Right, 23, 24 = Head
def visualize(image_bgr, mask, matrix, bbox_xywh, category, vis_way=2):
    color_mapping = {
        1: [255, 0, 0],  # 红色
        2: [0, 255, 0],  # 绿色
        3: [0, 0, 255],  # 蓝色
        4: [255, 255, 0],  # 黄色
        5: [255, 0, 255],  # 紫色
        6: [0, 255, 255],  # 青色
        7: [128, 0, 0],  # 深红色
        8: [0, 128, 0],  # 深绿色
        9: [0, 0, 128],  # 深蓝色
        10: [128, 128, 0],  # 深黄色
        11: [128, 0, 128],  # 深紫色
        12: [0, 128, 128],  # 深青色
        13: [255, 128, 0],  # 橙色
        14: [255, 0, 128],  # 粉红色
        15: [128, 255, 0],  # 浅黄色
        16: [128, 0, 255],  # 浅紫色
        17: [0, 255, 128],  # 浅青色
        18: [0, 128, 255],  # 天蓝色
        19: [255, 128, 128],  # 浅粉色
        20: [128, 255, 128],  # 淡绿色
        21: [128, 128, 255],  # 浅蓝色
        22: [192, 192, 192],  # 灰色
        23: [255, 255, 255],  # 白色
        24: [0, 0, 0]  # 黑色
        # 上面都是RGB，需要换成BGR
    }

    image_with_overlay = image_bgr.copy()
    x, y, w, h = [int(v) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return image_bgr

    # 假设你有一个颜色映射 cmap
    cmap = cv2.COLORMAP_JET  # 示例 cmap，你可以根据需要选择不同的颜色映射

    mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
    matrix_scaled = matrix.astype(np.float32) * 10  # 示例 val_scale
    _EPSILON = 1e-6
    if np.any(matrix_scaled > 255 + _EPSILON):
        print("Matrix has values > {255 + _EPSILON} after scaling, clipping to [0..255]")
    matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
    matrix_vis = cv2.applyColorMap(matrix_scaled_8u, cmap)

    if vis_way == 0:
        matrix_vis[mask_bg] = image_bgr[y: y + h, x: x + w, :][mask_bg]

    elif vis_way == 1:
        # 找到非零区域的像素，并将其设置为你想要的颜色
        nonzero_indices = np.where(mask != 0)
        matrix_vis[nonzero_indices] = [255, 0, 0]  # 示例设置为蓝色

    elif vis_way == 2:
        # 将整个 matrix_vis 设置为原始图像的颜色
        matrix_vis = image_bgr[y: y + h, x: x + w, :].copy()
        # for i in range(1, 24):  # 假设部位的标签从 1 到 24
        # 找到当前部位的像素索引
        part_indices = np.where(matrix == category)
        # 如果当前类别在颜色映射字典中有对应的颜色，则使用该颜色，否则随机生成一个颜色
        color = color_mapping.get(category, [np.random.randint(0, 256) for _ in range(3)])
        # 将当前部位的像素设置为指定颜色
        matrix_vis[part_indices] = color
    else:
        print("Error Visualization Way!")

    image_bgr[y: y + h, x: x + w, :] = (
            image_bgr[y: y + h, x: x + w, :] * (1.0 - 0.7) + matrix_vis * 0.7  # 示例 alpha
    )
    return image_bgr.astype(np.uint8), image_with_overlay


def draw_line_on_image(image, x, y):
    # 假设 x, y 是浮点数
    x = int(x)
    y = int(y)
    # 在图像上绘制线条
    cv2.line(image, (x, y - 30), (x, y + 30), (0, 255, 0), thickness=2)  # 在 y 坐标处画一条绿色的水平线

    return image


def visualize_seg_picture(height, width, results):
    # MagicAnimate uses the Viridis colormap for their training data
    cmap = cv2.COLORMAP_VIRIDIS

    # Visualizer outputs black for background, but we want the 0 value of
    # the colormap, so we initialize the array with that value
    arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
    out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)

    # 绘制图像
    plt.imshow(out_frame)
    plt.axis('on')  # 关闭坐标轴

    # 保存图片
    plt.savefig('example_image.jpg')  # 保存为 PNG 格式的图片，可以根据需要修改文件名和格式

    # 显示图像
    plt.show()


def print_iuv_to_txt(boxes_xywh, densepose_result):
    boxes_xywh = boxes_xywh.cpu().numpy()
    for i, result in enumerate(densepose_result):
        iuv_array = torch.cat(
            (result.labels[None].type(torch.float32), result.uv * 255.0)
        ).type(torch.uint8)
        iuv_array = iuv_array.cpu().numpy()  # 将 CUDA tensor 转换为 NumPy 数组
        print(iuv_array.shape)
        if boxes_xywh[i][2] <= 50 or boxes_xywh[i][3] <= 50:
            continue
        # 设置打印数组的行和列的最大数量
        np.set_printoptions(threshold=np.inf)

        # 这里是你的数组
        print("i: ")
        print(iuv_array[0])
        print("u: ")
        print(iuv_array[1])
        print("v: ")
        print(iuv_array[2])
        # 将数组写入txt文件
        np.savetxt('i.txt', iuv_array[0], fmt='%d')
        np.savetxt('u.txt', iuv_array[1], fmt='%d')
        np.savetxt('v.txt', iuv_array[2], fmt='%d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image_path", type=str, default="./input_image.jpg")
    args = parser.parse_args()

    main(args.input_image_path)
