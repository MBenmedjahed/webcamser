##from django.shortcuts import render

# Create your views here.
from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2

def index(request):
    return render(request, 'motion_detection/index.html')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.first_frame = None

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.first_frame is None:
            self.first_frame = gray
            return

        delta_frame = cv2.absdiff(self.first_frame, gray)
        thresh_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
