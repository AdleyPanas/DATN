import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv
import pyrealsense2 as rs
import math
import socket
import time
import threading
#Lấy thông số
#size=(1280,720)
size=(640,480)
config = rs.config()
config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8,30)
pipeline = rs.pipeline()
pipeline.start(config)
# Lấy profile
profile = pipeline.get_active_profile()
# Lấy thông số nội tại của dòng video màu
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cx=intr.ppx
cy=intr.ppy
fx=intr.fx
fy=intr.fy
pipeline.stop()
#khoi tao mo hinh YOLO v8
model = YOLO("yolov8n-face.pt","v8")
#khởi tạo bộ vẽ hộp bao quanh đối tượng
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
#khởi tạo camera độ sâu
class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
        self.pipeline.start(config)
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            return False, None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, depth_frame, color_image

    def release(self):
        self.pipeline.stop()
#đọc camera
dc = DepthCamera()
point = (400,300)
# #khởi tạo truyền thông
HOST = '0.0.0.0'  # Lắng nghe mọi IP
PORT = 12345      # Cổng kết nối
toa_do=(0,1,0)
#tạo vị trí chuột
# def show_distance(event, x, y, flags, param):
#     global point
#     point = (x,y)
# cv2.namedWindow("camera")
# cv2.setMouseCallback("camera",show_distance)
#nhận diện và theo dõi đối tượng trong ảnh
start_tracking = threading.Event()
def tracking_people():
    global toa_do
    start_tracking.wait()
    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        if not ret:
            continue
        results = model.track(source=color_frame, show=False, tracker='bytetrack.yaml', stream=True, device= "0")
        for result in results:
            detections = sv.Detections.from_yolov8(result)
            #kiem tra neu tra ve ID cua doi tuong, gan ID cho doi tuong
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            #chỉ lấy đối tượng là con người (ID=0), bỏ các đối tượng khác
            detections = detections[detections.class_id == 0]
            #kiểm tra nếu không có đối tượng thì hiển thị ảnh gốc
            if detections.xyxy.size == 0 or (detections.tracker_id is None or not detections.tracker_id.any()):
                cv2.imshow("camera", color_frame)
                toa_do=(0,1,0)
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
                    # x=point[0]
                    # y=point[1]
                    # print(point)
                    cv2.circle(color_frame,(x,y),5,(0,0,255),-1)
                    point=(x,y)
                    distance = depth_frame.get_distance(point[0],point[1])*1000
                    Xs=round(float(distance*(x-cx)/fx),0)
                    Ys=round(float(distance*(y-cy)/fy),0)
                    Zs=round(float(distance),0)
                    toa_do=(Xs,Ys,Zs)
                    cv2.putText(color_frame,f"{toa_do[0]:,.0f},{toa_do[1]:,.0f},{toa_do[2]:,.0f}", (point[0] + 70, point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (0, 255, 0), 2)
                    cv2.imshow("camera", color_frame)

        if cv2.waitKey(1) == ord(" "):
            break
def tcp_communication(conn, addr):
    print(f"[TCP] Bắt đầu xử lý {addr}")
    start_tracking.set()
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            message = data.decode().strip()
            #print(f"LabVIEW gửi: {message}")
            if message == "PING":
                data = f"{toa_do[0]},{toa_do[1]},{toa_do[2]},{time.time()}\r\n"
                conn.sendall(data.encode())
    except Exception as e:
        print(f"[TCP] Lỗi với {addr}: {e}")
    finally:
        conn.close()
        print(f"[TCP] Đã đóng kết nối với {addr}")
def tcp_server_thread():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print("Python server đang chờ LabVIEW kết nối...")
        while True:
            conn, addr = server.accept()
            print(f"Đã kết nối từ {addr}")
            tcp_thread = threading.Thread(target=tcp_communication, args=(conn, addr), daemon=True)
            tcp_thread.start()
if __name__ == "__main__":
    # Tạo các thread

    tracking_thread = threading.Thread(target=tracking_people,daemon=True)
    tracking_thread.start()
    start_tracking.set()

    #tcp_server = threading.Thread(target=tcp_server_thread, daemon=True)
    #tcp_server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Nhận tín hiệu dừng, thoát chương trình.")
dc.release()
cv2.destroyAllWindows()