from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os, threading, time, uuid, shutil, zipfile
from moduleAI import infer_folder
import uuid

CHUNK_SIZE = 5 * 1024 * 1024
MAX_PARALLEL = 10

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
PATCH_MODEL_1 = './model1.pth'
IS_POST = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=2_000_000_000)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

def clear_folder(folder):
    if os.path.exists(folder):
        for entry in os.listdir(folder):
            path = os.path.join(folder, entry)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)

def find_series(root_dir):
    series = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(fname.lower().endswith(".dcm") for fname in filenames):
            rel_path = os.path.relpath(dirpath, root_dir)
            name = os.path.basename(dirpath.rstrip(os.sep))
            if not name or name == os.curdir:
                if rel_path == os.curdir:
                    name = os.path.basename(root_dir.rstrip(os.sep)) or rel_path
                else:
                    name = os.path.basename(rel_path)
            series.append((name, dirpath))
    return series

def process_series(item_series_patch):
    index_to_russian = {
        0: "Нормальные случаи",
        1: "CAP (пневмония) случаи",
        2: "Случай COVID-19",
        3: "Случай рака"
    }
    df_results = infer_folder(item_series_patch, PATCH_MODEL_1)
    classIndex = df_results['pred_class'].iloc[0]
    prob = df_results.iloc[0, classIndex + 2]
    return classIndex, prob, index_to_russian[classIndex]

def process_zip(filename):
    temp_dir = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=True)
    zip_path = os.path.join(UPLOAD_FOLDER, filename)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    series_list = find_series(temp_dir)
    results = []
    for series_name, series_path in series_list:
        start = time.time()
        try:
            (classIndex, prob, index_to_russian) = process_series(series_path)
            end = time.time()
            results.append({
                "name": series_name,
                "duration": float(end - start),
                "model1":{
                    "classIndex": int(classIndex),
                    "prob": float(prob),
                    "index_to_russian": index_to_russian
                }
            })
        except Exception as e:
            results.append({
                "name": series_name,
                "error": str(e)
            })
            pass
    try:
        clear_folder(temp_dir)
        os.rmdir(temp_dir)
        os.remove(zip_path)
    except Exception as e:
        print(f"Ошибка удаления временных файлов: {e}")

    return results

# ================================ Загрузка через сокет ===============
tasks = {}

@socketio.on('start_upload')
def start_upload(data):
    """Инициализация задачи загрузки"""

    task_id = str(uuid.uuid4())
    unique_filename = f"{task_id}_{data['original_name']}"
    tasks[task_id] = {
        'chunks': set(),
        'total_chunks': data['total_chunks'],
        'filename': unique_filename,
        'sid': request.sid
    }

    emit('upload_started', {
        'task_id': task_id,
        'unique_name': unique_filename,
        'chunk_size': CHUNK_SIZE,
        'max_parallel': MAX_PARALLEL
    })

def socet_process_zip(task_id):
    task = tasks[task_id]
    filename = task['filename']
    sid = task['sid']
    result = process_zip(filename)
    socketio.emit('file_result', result, room=sid)
    del tasks[task_id]

@socketio.on('upload_chunk')
def handle_chunk(data):
    task_id = data['task_id']
    if task_id not in tasks:
        return

    task = tasks[task_id]
    file_path = os.path.join(UPLOAD_FOLDER, task['filename'])

    offset = data['offset']
    chunk_bytes = bytes(data['chunk'])

    with open(file_path, 'r+b' if os.path.exists(file_path) else 'wb') as f:
        f.seek(offset * CHUNK_SIZE)
        f.write(chunk_bytes)
    task['chunks'].add(offset)

    emit('chunk_received', {'task_id': task_id, 'offset': offset}, room=task['sid'])

    total_chunks = task['total_chunks']
    uploaded_chunks = len(task['chunks'])

    if uploaded_chunks == total_chunks:
        threading.Thread(target=socet_process_zip, args=(task_id, )).start()

# =================================== GET и POST запросы ===============================
@app.route('/')
def index():
    return render_template('index.html', IS_POST= 'true' if IS_POST else 'false')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    # Обработка файла
    results = process_zip(unique_filename)
    return jsonify(results)

if __name__ == "__main__":
    clear_folder(UPLOAD_FOLDER);
    clear_folder(TEMP_FOLDER);
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)