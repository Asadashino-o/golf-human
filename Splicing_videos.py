import cv2
import time
import numpy as np
from Batting_detection import getBattingFrame
from ultralytics import YOLO


# 如果返回-1 表示合成视频失败
# 只有返回0 才是正常合成视频成功 当然也不保证合成的视频一定是正确的
def mergeFrontAndSideVideo(front_video, side_video, model, output_video):
    # 获得击球帧
    batting_frame_start_time = time.time()
    front_batting_frame = getBattingFrame(model, front_video)
    side_batting_frame = getBattingFrame(model, side_video)

    if front_batting_frame == -1 or side_batting_frame == -1:
        return -1

    print(f"batting_time:{time.time() - batting_frame_start_time}")
    print(f"front_batting_frame:{front_batting_frame}")
    print(f"side_batting_frame:{side_batting_frame}")

    # 获得两个视频的所有帧的时间，按照击球帧那一帧为10000ms为绝对时间戳进行计算
    front_video_all_time = getVideoAllTime(front_video, front_batting_frame)
    side_video_all_time = getVideoAllTime(side_video, side_batting_frame)

    video_dict = getCorr(front_video_all_time, side_video_all_time)

    front_video_cap = cv2.VideoCapture(front_video)
    side_video_cap = cv2.VideoCapture(side_video)
    video1_frame_dict = dict()
    video2_frame_dict = dict()
    index = 0
    while True:
        ret, frame = front_video_cap.read()
        if not ret:
            break
        video1_frame_dict[index] = frame
        index += 1
    index = 0
    while True:
        ret, frame = side_video_cap.read()
        if not ret:
            break
        video2_frame_dict[index] = frame
        index += 1

    output_fps = 30
    video1_width = int(front_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video1_height = int(front_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video2_width = int(side_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video2_height = int(side_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需要更改编码器

    out = cv2.VideoWriter(output_video, fourcc, output_fps,
                          (video1_width + video2_width, max(video1_height, video2_height)))
    output_index = 0
    output_batting_index = -1
    for index in range(len(front_video_all_time)):
        if index in video_dict:
            if front_video_all_time[index] == 10000:
                output_batting_index = output_index
            # print(index, video_dict[index])
            result = np.concatenate((video1_frame_dict[index], video2_frame_dict[video_dict[index]]), axis=1)
            out.write(result)
            output_index += 1
    out.release()
    front_video_cap.release()
    side_video_cap.release()

    return output_batting_index


# 按照指定帧为10000ms,计算其他帧的相对值
def getVideoAllTime(video, frame_num):
    # video_first_time = getVideoFirstTime(video,frame_num)
    cap = cv2.VideoCapture(video)
    dura = 0
    all_time = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dura = cap.get(cv2.CAP_PROP_POS_MSEC)
        all_time.append(dura)
    diff = 10000 - all_time[frame_num]
    res_time = [i + diff for i in all_time]
    return res_time


def getCorr(video1_time, video2_time):
    video_dict = dict()
    minn = 10  # ms
    for index1 in range(len(video1_time)):
        index1_minn = minn
        for index2 in range(len(video2_time)):
            if abs(video1_time[index1] - video2_time[index2]) < index1_minn:
                video_dict[index1] = index2
                index1_minn = abs(video1_time[index1] - video2_time[index2])
    return video_dict


if __name__ == '__main__':
    model_file = "ballAndClub.pt"
    model = YOLO(model_file)
    front_video_path = "./sample_videos/fo1.mp4"
    side_video_path = "./sample_videos/side1.mp4"
    output_video_path = "./sample_videos/merge1.mp4"
    batting_frame = mergeFrontAndSideVideo(front_video_path, side_video_path, model, output_video_path)
    print(batting_frame)
