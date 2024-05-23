import os
import re
import threading
import sys
import time
from flask_cors import CORS
from flask import Flask, render_template, request, redirect, abort, jsonify, send_file
from moviepy.video.io.VideoFileClip import VideoFileClip

sys.path.append('..')  # 将上级目录添加到Python路径中
from merge2 import main as both
from head_and_buttock import main as dtl
from head import head as fo

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

PROCESSED_VIDEO_DIR = os.path.join(os.getcwd(), 'processed')
OUTPUT = os.path.join(os.getcwd(), 'processed/output')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


def process_video(filepath, processed_filepath, process_id, detection_type):
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
    # 根据检测类型选择不同的处理逻辑
    if detection_type == 'Face-on':
        fo(filepath, processed_filepath, number, 4)
    elif detection_type == 'Down-the-line':
        dtl(filepath, processed_filepath, number, 4)
    elif detection_type == 'Both':
        both(filepath, processed_filepath, number, 4)

    # 更新进度为100%
    progress_dict[process_id] = 100
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    runtime = end_time - start_time

    print("程序运行时间为：", runtime, "秒")


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

        detection_type = request.form.get('detection')
        if detection_type not in ['Face-on', 'Down-the-line', 'Both']:
            return jsonify({'error': 'Invalid detection type'}), 400

        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        process_id = filename

        progress_dict[process_id] = 0
        thread = threading.Thread(target=process_video, args=(filepath, processed_filepath, process_id, detection_type))
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
        if os.path.exists(file_path):
            reduce_bitrate(file_path, output)
            return send_file(output, mimetype='video/mp4')  # 使用 send_file 发送文件
        else:
            return abort(404, description="File not found")
    except Exception as e:
        return abort(500, description=str(e))


def reduce_bitrate(input_video, output_video):
    video_clip = VideoFileClip(input_video)
    video_clip.write_videofile(output_video, codec="libx264")  # 将视频编码改为h264，便于在页面展示
    video_clip.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2348, debug=True) # 这是服务器的端口设置，目前是全开放端口,仅供测试并不安全，运行后的网址是{your_server_name}:2348
