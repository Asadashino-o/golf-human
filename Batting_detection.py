import cv2
from ultralytics import YOLO


# 获得一个视频文件的击球帧号和球位置，索引从0开始
def getBattingFrameAndPosition(model, video_file):
    ball_list = getModelBall(model, video_file)

    if len(ball_list) == 0:
        return -1, None

    diff_list = detectFrameWithVideoRectList(video_file, ball_list)
    batting_frame_list = []
    ball_positions = []
    for ele_list in diff_list:
        for ele in ele_list:
            if ele["diff"] > 0.12:
                print(ele["diff"], ele["startFrameNum"])
                batting_frame_list.append(ele["startFrameNum"])
                ball_positions.append(ele["ball_position"])  # 返回球的位置坐标而不是索引
    if len(batting_frame_list) == 0:
        return -1, None
    print(batting_frame_list)
    return batting_frame_list[-1], ball_positions[-1]


# 这个方法并不保证一定能找到高尔夫球
def getModelBall(model, video_file):
    cap = cv2.VideoCapture(video_file)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_total_frame = int(total_frame / 2)
    frame = None
    while half_total_frame != 0:
        ret, frame = cap.read()  # 偶数
        half_total_frame -= 1
    cap.release()
    model_res = modelDetect(model, frame)
    ball_list = model_res["ball_list"]
    if len(ball_list) > 0:
        return model_res["ball_list"]

    # 此时如果这张图片不存在高尔夫球，则直接按照按照20帧的方式寻找高尔夫球
    cap = cv2.VideoCapture(video_file)

    while True:
        count = 20
        ret = True
        while count > 0:
            ret, frame = cap.read()
            count -= 1
            if not ret:
                break
        if not ret:
            break
        model_res = modelDetect(model, frame)
        ball_list = model_res["ball_list"]
        if len(ball_list) > 0:
            break

    return ball_list


# 返回高尔夫球的位置
def modelDetect(model, pic):
    result = model(pic)[0]

    res = dict()
    ball_list = []
    for index in range(len(result.boxes.cls)):
        if result.boxes.conf[index] > 0.4 and result.boxes.cls[index] == 0:
            boxes = result.boxes.xywh[index].tolist()
            x = boxes[0] - boxes[2] / 2
            y = boxes[1] - boxes[3] / 2
            ball_list.append(((int(x), int(y)), (int(boxes[2]), int(boxes[3]))))
    res["ball_list"] = ball_list
    return res


# 检测视频中rect_list位置的帧阈值返回
# 返回格式[[ball1_threshold, ball2_threshold],....]
# 返回是列表，列表的每一个元素也是一个列表表示一帧之间的帧差，这个列表存储的是每个rect的帧差
def detectFrameWithVideoRectList(videoFilePath, rect_list):
    cap = cv2.VideoCapture(videoFilePath)
    if not cap.isOpened():
        return -1
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    frameNum = 0  # 表示帧的数目
    results = []
    while ret:
        # 求的是绝对值
        results.append([])
        for index in range(len(rect_list)):
            rect = rect_list[index]
            diff = getDiffWithAbs(frame1, frame2, rect)
            dic = dict()
            dic["startFrameNum"] = frameNum  # 从0开始，比较的第一帧
            dic["ball_position"] = rect[0]  # 返回球的位置坐标
            dic["diff"] = diff / (rect[1][0] * rect[1][1] * 255 * 3)
            results[-1].append(dic)
        frame1 = frame2
        ret, frame2 = cap.read()
        frameNum += 1
    return results


# 输入两帧和一个矩形框，返回两帧之间的帧差 左闭右开
# 帧差是绝对值
def getDiffWithAbs(frame1, frame2, rect):
    diff = 0
    for j in range(rect[0][0], rect[0][0] + rect[1][0], 1):
        for i in range(rect[0][1], rect[0][1] + rect[1][1], 1):
            for k in range(3):
                diff += abs(int(frame2[i][j][k]) - int(frame1[i][j][k]))
    return diff


def get_frame_and_position(input_video_path, model_file):
    model = YOLO(model_file)
    batting_frame, ball_position = getBattingFrameAndPosition(model, input_video_path)
    return batting_frame, ball_position


if __name__ == '__main__':
    model_file = "ballAndClub.pt"
    model = YOLO(model_file)
    input_video_path = "./sample_videos/214.mp4"
    batting_frame, ball_position = getBattingFrameAndPosition(model, input_video_path)
    print("Batting Frame:", batting_frame)
    print("Ball Position:", ball_position)
