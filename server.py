from flask import Flask, request, jsonify, send_from_directory
import subprocess
import threading
import queue
import time
import os
import main

output_history = []
history_lock = threading.Lock()
current_inventory = []

app = Flask(__name__, static_folder='static', static_url_path='')

game_proc = None
output_queue = queue.Queue()

def start_game():
    global game_proc, current_inventory
    if game_proc is not None:
        return
    # Start with an empty inventory and let the game process
    # send the up–to–date list via stdout
    current_inventory = []
    game_proc = subprocess.Popen(
        ['python', '-u', 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    threading.Thread(target=_reader, daemon=True).start()

def _reader():
    global current_inventory
    for line in game_proc.stdout:
        if main.INVENTORY_PREFIX in line:
            data = line.split(main.INVENTORY_PREFIX,1)[1].strip()
            items = data.split('|') if data else []
            with history_lock:
                current_inventory = [i.strip() for i in items if i.strip()]
            continue
        output_queue.put(line)
        with history_lock:
            output_history.append(line)

def _get_output():
    lines = []
    while not output_queue.empty():
        lines.append(output_queue.get())
    return ''.join(lines)

@app.route('/api/start')
def api_start():
    start_game()
    time.sleep(1)
    intro = _get_output()
    status = _status()
    return jsonify({'intro': intro, **status})

@app.route('/api/command', methods=['POST'])
def api_command():
    cmd = request.json.get('command', '')
    if game_proc is None:
        return jsonify({'output': 'Game not running.'})
    game_proc.stdin.write(cmd + '\n')
    game_proc.stdin.flush()
    time.sleep(1)
    output = _get_output()
    status = _status()
    return jsonify({'output': output, **status})

@app.route('/api/poll')
def api_poll():
    return jsonify({'output': _get_output()})

@app.route('/api/history')
def api_history():
    with history_lock:
        history_text = ''.join(output_history)
    return jsonify({'history': history_text, **_status()})

@app.route('/api/status')
def api_status():
    return jsonify(_status())

def _status():
    hp = main.get_current_hp()
    with history_lock:
        inv = list(current_inventory)
    return {'hp': {'current': hp, 'max': 10}, 'inventory': inv}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)
