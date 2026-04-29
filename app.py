from flask import Flask, render_template, request
import os
import sys
import subprocess

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'quantum_tcd_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/batch', methods=['POST'])
def batch_inference():
    if 'video' not in request.files:
        return "No file selected", 400

    file = request.files['video']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    try:
        # Run inference
        cmd = [sys.executable, os.path.join(ROOT_DIR, "module9", "inference.py"), temp_path]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if result.returncode == 0:
            # Extract verdict from last line of stdout
            lines = result.stdout.strip().split('\n')
            verdict = lines[-1].strip().lower() if lines else 'unknown'
            return f"Analysis completed! Video classified as: {verdict}. Check module9/reports/ for detailed results.", 200
        else:
            return f"Analysis failed: {result.stderr}", 500

    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return f"Error running inference: {str(e)}", 500

@app.route('/realtime')
def realtime():
    try:
        # Run real-time detection in a separate thread to avoid blocking
        import threading
        def run_realtime():
            cmd = [sys.executable, os.path.join(ROOT_DIR, "module10", "realtime.py")]
            subprocess.run(cmd, cwd=ROOT_DIR)

        thread = threading.Thread(target=run_realtime, daemon=True)
        thread.start()

        return "Real-time detection started! Camera window should open.", 200
    except Exception as e:
        return f"Error starting real-time detection: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)