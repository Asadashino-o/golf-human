import argparse
import os
import torch
from torch import nn
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.export import STABLE_ONNX_OPSET_VERSION
from detectron2.modeling import build_model
from densepose import add_densepose_config
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger


def setup_cfg(args):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg


class DensePoseTracingAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        with torch.no_grad():
            if images.ndim == 3:
                images = images.unsqueeze(0)
            elif images.ndim != 4:
                raise ValueError(f"Expected input tensor to have 4 dimensions (N, C, H, W), but got {images.ndim}")

            outputs = self.model([{"image": images[0]}])
            instances = outputs[0]["instances"]

            scores = instances.scores
            pred_boxes = instances.pred_boxes.tensor
            pred_densepose = instances.pred_densepose

            densepose_results, boxes_xywh = self.extract_densepose_results(pred_densepose, pred_boxes, scores)
            return densepose_results, boxes_xywh

    def extract_densepose_results(self, pred_densepose, pred_boxes, scores):
        # 确保 pred_densepose 是 DensePoseChartPredictorOutputWithConfidences 类实例
        if not hasattr(pred_densepose, "coarse_segm") or not hasattr(pred_densepose, "u") or not hasattr(pred_densepose,
                                                                                                         "v"):
            raise AttributeError("pred_densepose does not have the required attributes.")

        dp_x = pred_densepose.coarse_segm
        dp_u = pred_densepose.u
        dp_v = pred_densepose.v

        densepose_results = list(zip(dp_x, dp_u, dp_v))
        boxes_xywh = pred_boxes

        filtered_densepose_results = []
        filtered_boxes_xywh = []
        for i, score in enumerate(scores):
            if score > 0.95:
                filtered_densepose_results.append(densepose_results[i])
                filtered_boxes_xywh.append(boxes_xywh[i])

        if not filtered_boxes_xywh:
            return None, None

        max_area_box = max(zip(filtered_densepose_results, filtered_boxes_xywh), key=lambda x: x[1][2] * x[1][3])
        return [max_area_box[0]], [max_area_box[1]]


def export_tracing(torch_model, inputs):
    image = inputs[0]["image"].to("cuda")
    inputs = [{"image": image}]

    traceable_model = DensePoseTracingAdapter(torch_model)
    traceable_model.eval()
    densepose_results, boxes_xywh = traceable_model(image.unsqueeze(0))

    print("DensePose Results: ", densepose_results)
    print("Bounding Boxes: ", boxes_xywh)

    if args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image.unsqueeze(0),), f, opset_version=STABLE_ONNX_OPSET_VERSION)
    logger.info("ONNX export complete.")


def get_sample_inputs(args):
    if args.sample_image is None:
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        original_image = detection_utils.read_image(args.sample_image, format=cfg.INPUT.FORMAT)
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return [inputs]


def main() -> None:
    global logger, cfg, args
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument("--format", choices=["onnx"], help="output format", default="onnx")
    parser.add_argument("--export-method", choices=["tracing"], help="Method to export models", default="tracing")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args)
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    sample_inputs = get_sample_inputs(args)
    if args.export_method == "tracing":
        export_tracing(torch_model, sample_inputs)

    if args.run_eval:
        logger.info("Evaluation not implemented in this example.")

    logger.info("Success.")


if __name__ == "__main__":
    main()
