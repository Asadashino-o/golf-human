import argparse
from head import head


def main(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4"):
    head(input_video_path, output_video_path, 1000, 1)  # 1000 is the number of detection frames


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
