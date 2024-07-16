import os
import threading
import time
from flask_cors import CORS
from flask import Flask, render_template, request, redirect, abort, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from Batting_detection import get_frame_and_position
from Splicing_videos import mergeFrontAndSideVideo
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('try.html')


def process_video(filepath, processed_filepath, process_id, detection_type, message=""):
    start_time = time.time()
    global number

    model_file = "ballAndClub.pt"
    number, ball_position = get_frame_and_position(filepath, model_file)
    if number == -1:
        progress_dict[process_id] = 1
        print("404 not found")
        return
    progress_dict[process_id] = 30  # 更新进度为30%
    print("击球帧为:", number)
    print("被击打的球在:", ball_position)

    if detection_type == 'Face-on':
        message = fo(filepath, processed_filepath, number, 4, ball_position)
    elif detection_type == 'Down-the-line':
        message = dtl(filepath, processed_filepath, number, 4, ball_position)
    elif detection_type == 'Both':
        message = both(filepath, processed_filepath, number, 4)

    end_time = time.time()
    runtime = end_time - start_time
    runtime_message = f"检测时间为：{runtime} 秒"
    print(runtime_message)

    full_message = message + "\n" + runtime_message

    txt_filename = os.path.splitext(processed_filepath)[0] + '.txt'
    with open(txt_filename, 'w') as f:
        f.write(full_message)

    # 更新进度为100%
    progress_dict[process_id] = 100


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
    file_path = os.path.join(PROCESSED_VIDEO_DIR, filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='video/mp4')
        else:
            return abort(404, description="File not found")
    except Exception as e:
        return abort(500, description=str(e))


@app.route('/processed_txt/<filename>', methods=['GET'])
def processed_txt(filename):
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    file_path = os.path.join(PROCESSED_VIDEO_DIR, txt_filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='text/plain')
        else:
            return abort(404, description="File not found")
    except Exception as e:
        return abort(500, description=str(e))


def process_videos(front_filepath, side_filepath, processed_filepath, process_id):
    model_file = "ballAndClub.pt"
    try:
        model = YOLO(model_file)
        progress_dict[process_id] = 20
        result = mergeFrontAndSideVideo(front_filepath, side_filepath, model, processed_filepath)
        if result == -1:
            progress_dict[process_id] = 1
            print("404 not found")
            return
        # 更新进度为100%
        else:
            progress_dict[process_id] = 100
    except Exception as e:
        print(f"Error processing videos: {e}")
        progress_dict[process_id] = 0  # 处理失败


@app.route('/upload-multiple', methods=['POST'])
def upload_multiple_files():
    if 'file-front' not in request.files or 'file-side' not in request.files:
        return redirect(request.url)
    file_front = request.files['file-front']
    file_side = request.files['file-side']
    if file_front.filename == '' or file_side.filename == '':
        return redirect(request.url)
    if allowed_file(file_front.filename) and allowed_file(file_side.filename):
        filename_front = file_front.filename
        filename_side = file_side.filename
        filepath_front = os.path.join(app.config['UPLOAD_FOLDER'], filename_front)
        filepath_side = os.path.join(app.config['UPLOAD_FOLDER'], filename_side)
        file_front.save(filepath_front)
        file_side.save(filepath_side)

        processed_filename = f"merged_{filename_front}_{filename_side}"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        process_id = f"{filename_front}_{filename_side}"

        progress_dict[process_id] = 0
        thread = threading.Thread(target=process_videos,
                                  args=(filepath_front, filepath_side, processed_filepath, process_id))
        thread.start()

        return jsonify({'process_id': process_id, 'filename': processed_filename})
    return redirect(request.url)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=20001, debug=True)
