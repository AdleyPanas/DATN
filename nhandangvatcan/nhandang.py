import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv
import time
import threading
import os
from frame import *
#Lấy thông số
#size=(1280,720)
size=(1280,720)
#khoi tao mo hinh YOLO v8
model = YOLO(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "yolov8n-face.pt")),"v8")
model_2 = YOLO(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "yolov8m.pt")),"v8")
#print(model_2.model.names)
#khởi tạo bộ vẽ hộp bao quanh đối tượng
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

#đọc camera
dc = DepthCamera()
point = (400,300)
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

    def get_distance(map, x, y):
        try:
            return map[y, x]
        except:
            print("x={} y= {}".format(x, y))
            return map[y, x]
    def tinh_toado(x,y,distance,cx,cy,fx,fy):
        if distance <=0:
            return (0,0,0)
        Xs = distance*(x-cx)/fx
        Ys = distance*(y-cy)/fy
        Zs = distance
        return (Xs,Ys,Zs)
    while True:
        ret, depth_frame, color_frame, intrinsics = dc.get_frame()
        cx = intrinsics.ppx
        cy = intrinsics.ppy
        fx = intrinsics.fx
        fy = intrinsics.fy
        tam = (int(cx), int(cy))
        if not ret:
            continue
        results = model.track(source=color_frame, show=False, tracker='bytetrack.yaml', stream=True, device= "0")
        results_2 = model_2.track(source=color_frame, show=False, tracker='bytetrack.yaml', stream=True, device="0")
        for result,result_2 in zip(results,results_2):
            detections = sv.Detections.from_yolov8(result)
            obstacles  = sv.Detections.from_yolov8(result_2)
            #kiem tra neu tra ve ID cua doi tuong, gan ID cho doi tuong
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            if result_2.boxes.id is not None:
                obstacles.tracker_id = result_2.boxes.id.cpu().numpy().astype(int)

            #obstacles = obstacles[obstacles.class_id != 0]
            # obstacles = obstacles[
            #     (obstacles.class_id != 0) & ~((obstacles.class_id >= 30) & (obstacles.class_id <= 55))
            #                                   & ~((obstacles.class_id >= 64) & (obstacles.class_id <= 79))]
            obstacles = obstacles[(obstacles.class_id == 39)]
            #kiểm tra nếu không có đối tượng thì hiển thị ảnh gốc
            if detections is None or detections.tracker_id is None or detections.class_id is None or detections.confidence is None or \
                    obstacles is None or obstacles.tracker_id is None or obstacles.class_id is None or obstacles.confidence is None:
                cv2.imshow("camera", color_frame)
                toa_do=(0,1,0)
            else:
                #nếu có đối tượng thì vẽ box lên hình
                labels = [
                    f'id: {tr_id} {model.model.names[int(cls_id)]} {conf:.2f}'
                    for tr_id, cls_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
                ]
                labels_2 = [
                    f'id: {tr2_id} {model_2.model.names[int(cls2_id)]} {conf2:.2f}'
                    for tr2_id, cls2_id, conf2 in zip(obstacles.tracker_id, obstacles.class_id, obstacles.confidence)
                ]
                for box_2 in obstacles.xyxy:
                    xa, ya, xb, yb = box_2.astype(int)
                    #xác định tâm vật cản
                    center_x = (xa+xb)//2
                    center_y = (ya+yb)//2
                    depth_obs = get_distance(depth_frame, center_x, center_y)
                    #tính tọa độ vật cản
                    toa_do_obs = tinh_toado(center_x,center_y,depth_obs,cx,cy, fx, fy)
                    cv2.putText(color_frame, f"{toa_do_obs[0]:,.0f},{toa_do_obs[1]:,.0f},{toa_do_obs[2]:,.0f}",
                                (center_x + 70, center_y - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 255, 0), 2)
                color_frame = box_annotator.annotate(scene=color_frame, detections=detections, labels=labels)
                color_frame = box_annotator.annotate(scene=color_frame, detections=obstacles, labels=labels_2)

                # lấy đối tượng có ID = 1
                xyxy = detections.xyxy[detections.tracker_id == 1]
                # xác định tâm đối tượng và vẽ tâm
                if np.array(xyxy).size != 0:
                    x1,y1,x2,y2 = np.array(xyxy.astype(int))[0]
                    x=(x1+x2)//2
                    y=(y1+y2)//2
                    cv2.circle(color_frame,(x,y),5,(0,0,255),-1)
                    point=(x,y)
                    distance = get_distance(depth_frame,point[0],point[1])
                    toa_do=tinh_toado(x,y,distance, cx, cy, fx, fy)
                    cv2.putText(color_frame,f"{toa_do[0]:,.0f},{toa_do[1]:,.0f},{toa_do[2]:,.0f}", (point[0] + 70, point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (0, 255, 0), 2)
                    cv2.imshow("camera", color_frame)

        if cv2.waitKey(1) == ord(" "):
            break

if __name__ == "__main__":
    # Tạo các thread

    tracking_thread = threading.Thread(target=tracking_people,daemon=True)
    tracking_thread.start()
    start_tracking.set()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] Nhận tín hiệu dừng, thoát chương trình.")
dc.release()
cv2.destroyAllWindows()