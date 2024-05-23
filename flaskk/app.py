import os
import re
import threading
import sys
import time
from flask_cors import CORS

from flask import Flask, render_template, request, redirect, abort, jsonify, send_file
from moviepy.video.io.VideoFileClip import VideoFileClip

sys.path.append('..')  # 将上级目录添加到Python路径中
from merge2 import main as fct  # 替换 your_function 为你在 merge2.py 中想要导入的函数名或变量名
from queue import Queue

# 创建一个全局队列用于线程间通信
message_queue = Queue()
app = Flask(__name__)
CORS(app)  # 允许跨域请求
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 设置最大文件大小为100MB

progress_dict = {}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])
# 假设处理后的视频存储在项目根目录下的 processed_videos 目录中
PROCESSED_VIDEO_DIR = os.path.join(os.getcwd(), 'processed')
OUTPUT = os.path.join(os.getcwd(), 'processed/output')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


def process_video(filepath, processed_filepath, process_id):
    # 记录开始时间
    start_time = time.time()
    global number
    filename = os.path.basename(filepath)
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r'(\d+)', filename)
    if match:
        number_str = match.group(1)  # 获取匹配到的第一个数字字符串
        number = int(number_str)
        print("提取的数字:", number)
    else:
        print("未找到数字")
    fct(filepath, processed_filepath, number, 4)
    # 更新进度为100%
    progress_dict[process_id] = 100
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    runtime = end_time - start_time

    print("程序运行时间为：", runtime, "秒")
    # message += "程序运行时间为：{} 秒".format(runtime)
    # 将消息放入队列中
    # message_queue.put(message)


# @app.route('/get_message', methods=['GET'])
# def get_message():
#     # 从队列中获取消息，并返回给客户端
#     message = message_queue.get()
#     return jsonify({'message': message})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the video in a separate thread
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        process_id = filename

        progress_dict[process_id] = 0
        thread = threading.Thread(target=process_video, args=(filepath, processed_filepath, process_id))
        thread.start()

        return jsonify({'process_id': process_id, 'filename': processed_filename})
    return redirect(request.url)


@app.route('/progress/<process_id>')
def progress(process_id):
    progress = progress_dict.get(process_id, 0)
    status = 'completed' if progress == 100 else 'processing'
    return jsonify({'progress': progress, 'status': status})


@app.route('/processed/<filename>', methods=['GET'])
def processed(filename):
    # 生成处理后的视频文件的绝对路径
    file_path = os.path.join(PROCESSED_VIDEO_DIR, filename)
    output = os.path.join(OUTPUT, filename)
    try:
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 使用 send_file 发送文件
            reduce_bitrate(file_path, output)
            return send_file(output, mimetype='video/mp4')
        else:
            # 文件不存在，返回404错误
            return abort(404, description="File not found")
    except Exception as e:
        # 处理其他可能的异常
        return abort(500, description=str(e))


def reduce_bitrate(input_video, output_video):
    # 读取输入视频文件
    video_clip = VideoFileClip(input_video)

    # 改为h264
    video_clip.write_videofile(output_video, codec="libx264")

    # 关闭视频文件对象
    video_clip.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2348, debug=True)
