import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv
import pyrealsense2 as rs
import math
import socket
import time

#khoi tao mo hinh YOLO v8
model = YOLO("yolov8m.pt","v8")
#khởi tạo bộ vẽ hộp bao quanh đối tượng
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
#khởi tạo camera độ sâu
class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()
#đọc camera
dc = DepthCamera()
point = (400,300)
tam = 320
#khởi tạo truyền thông
HOST = "192.168.0.107"  # Địa chỉ IP của LabVIEW
PORT = 6000  # Port phải khớp với LabVIEW
# Tạo socket TCP
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Kết nối đến LabVIEW
client.connect((HOST, PORT))
print("Đã kết nối đến LabVIEW!")
#nhận diện và theo dõi đối tượng trong ảnh
while True:
    ret, depth_frame, color_frame = dc.get_frame()
    if not ret:
        break
    results = model.track(source=color_frame, show=False, tracker='bytetrack.yaml', stream=True, device= "0")
    for result in results:
        detections = sv.Detections.from_yolov8(result)
        #kiem tra neu tra ve ID cua doi tuong, gan ID cho doi tuong
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        #chỉ lấy đối tượng là con người (ID=0), bỏ các đối tượng khác
        detections = detections[detections.class_id == 0]
        #kiểm tra nếu không có đối tượng thì hiển thị ảnh gốc
        if detections.xyxy.size == 0:
            cv2.imshow("camera", color_frame)
        else:
            #nếu có đối tượng thì vẽ box lên hình
            labels = [
                f'id: {tr_id} {model.model.names[int(cls_id)]} {conf:.2f}'
                for tr_id, cls_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]
            color_frame = box_annotator.annotate(scene=color_frame, detections=detections, labels=labels)
            # lấy đối tượng có ID = 1
            xyxy = detections.xyxy[detections.tracker_id == 1]
            # xác định tâm đối tượng và vẽ tâm
            if np.array(xyxy).size != 0:
                x1,y1,x2,y2 = np.array(xyxy.astype(int))[0]
                x=(x1+x2)//2
                y=(y1+y2)//2
                cv2.circle(color_frame,(x,y),5,(0,0,255),-1)
                point=(x,y)
                distance = depth_frame[point[1], point[0]] / 1000
                goc = round(math.atan((tam - point[0]) / 606), 3)
                cv2.putText(color_frame, "{}m".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2)
                cv2.putText(color_frame, "{}do".format(goc), (point[0] + 70, point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2)
            else:
                point=()
                distance = 0
                goc = 0
            #thuật toán tính vị trí đặt của xe
            xN=distance
            yN=distance*math.tan(goc) #tọa độ người
            xP = round(xN - (1+0.17)*math.cos(goc),3)
            yP = round(yN - (1+0.17)*math.sin(goc),3)
            data = f"{xP},{yP},{goc},1\r\n"
            client.sendall(data.encode())  # Chuyển chuỗi thành bytes
            data2 = f"{xP},{yP},{goc},0\r\n"
            client.sendall(data2.encode())
            cv2.imshow("camera",color_frame)
            time.sleep(0.1)
    if cv2.waitKey(1) == ord(" "):
        break
dc.release()
client.close()
cv2.destroyAllWindows()