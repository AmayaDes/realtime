from flask import Flask, render_template, Response, jsonify
import cv2
import real_measurements
import webbrowser
import threading


app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(real_measurements.height_cal(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/send-data',methods=['GET', 'POST'])
def send_data():
    data = {
        'shoulder_distance_cm': '%.2f' % real_measurements.shoulder_distance_inch,
        'hip_distance_cm': '%.2f' % real_measurements.hip_distance_inch,
        'height_cm': '%.2f' % real_measurements.get_height(),
        'size': real_measurements.get_size()
    }
    return jsonify(data)

@app.route('/run_python_script', methods=['POST'])
def run_python_script():
    # Execute the Python script
    return 'Python script executed'

def open_browser():
    webbrowser.open_new_tab('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True)

