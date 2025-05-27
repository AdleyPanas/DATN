import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import threading
import os
from frame import *
class PeopleTracker:
    def __init__(self,show_display=True):
        # Khởi tạo model và các biến cần thiết
        # Model phát hiện người
        self.model = YOLO('model/yolov8n-face.pt')
        #Model phát hiện vật cản
        self.model_2 = YOLO('model/yolov8m.pt')
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        self.toa_do_nguoi = (0, 0, 0)
        self.toa_do_vat_can = []
        self.running = False
        self.lock = threading.Lock()
        self.dc = DepthCamera()
        self.show_display = show_display
        self.latest_display_frame = None

    def get_distance(self, depth_map, x, y):
        try:
            return depth_map[y, x]
        except:
            print(f"Index out of bounds: x={x} y={y}")
            return 0

    def tinh_toado(self, x, y, distance, cx, cy, fx, fy):
        if distance <= 0:
            return (0, 0, 0)
        Xs = distance * (x - cx) / fx
        Ys = distance * (y - cy) / fy
        Zs = distance
        return (Xs, Ys, Zs)
    def stop_tracking(self):
        self.running = False
        try:
            self.dc.release()  # Chỉ gọi nếu đã start
        except RuntimeError as e:
            print("Lỗi dừng DepthCamera:", e)
        cv2.destroyAllWindows()
        print('Đã dừng camera và giải phóng tài nguyên')

    def start_tracking(self):
        self.running = True
        self.thread = threading.Thread(target=self.tracking_people)
        self.thread.start()
    def tracking_people(self):
        self.running = True
        try:
            while self.running:
                ret, depth_frame, color_frame, intrinsics = self.dc.get_frame()
                if not ret:
                    continue
                cx, cy = intrinsics.ppx, intrinsics.ppy
                fx, fy = intrinsics.fx, intrinsics.fy
                # Phát hiện người và vật cản
                results = self.model.predict(source=color_frame, stream=False, device="0")
                results_2 = self.model_2.predict(source=color_frame, stream=False,device="0")

                for result, result_2 in zip(results, results_2):
                    detections = sv.Detections.from_yolov8(result)
                    obstacles = sv.Detections.from_yolov8(result_2)

                    # Xử lý ID tracker
                    # if result.boxes.id is not None:
                    #     detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    # if result_2.boxes.id is not None:
                    #     obstacles.tracker_id = result_2.boxes.id.cpu().numpy().astype(int)
                    detections.tracker_id = None
                    obstacles.tracker_id = None
                    # Lọc chỉ lấy vật cản class 39 (tùy chỉnh theo model của bạn)
                    obstacles = obstacles[(obstacles.class_id == 39)]

                    with self.lock:
                        # Cập nhật tọa độ vật cản
                        self.toa_do_vat_can = []
                        if len(obstacles.xyxy)>0:
                            for box_2 in obstacles.xyxy:
                                xa, ya, xb, yb = box_2.astype(int)
                                center_x = (xa + xb) // 2
                                center_y = (ya + yb) // 2
                                depth_obs = self.get_distance(depth_frame, center_x, center_y)
                                toa_do_obs = self.tinh_toado(center_x, center_y, depth_obs, cx, cy, fx, fy)
                                self.toa_do_vat_can.append(toa_do_obs)
                                #Vẽ thông tin vật cản lên frame
                                cv2.putText(color_frame,
                                            f"{toa_do_obs[0]:.0f},{toa_do_obs[1]:.0f},{toa_do_obs[2]:.0f}mm",
                                            (center_x + 20, center_y - 10),cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0), 2)
                        # Cập nhật tọa độ người
                        closest_distance = float('inf')
                        closest_person_coord = (0, 0, 0)
                        for box in detections.xyxy:
                            x1, y1, x2, y2 = box.astype(int)
                            x = (x1 + x2) // 2
                            y = (y1 + y2) // 2
                            distance = self.get_distance(depth_frame, x, y)

                            if 0 < distance < closest_distance:
                                closest_distance = distance
                                closest_person_coord = self.tinh_toado(x, y, distance, cx, cy, fx, fy)
                                best_x, best_y = x, y  # Lưu để vẽ lên frame
                        self.toa_do_nguoi = closest_person_coord
                        if closest_distance < float('inf'):
                            cv2.circle(color_frame, (best_x, best_y), 5, (0, 0, 255), -1)
                            cv2.putText(color_frame,
                                        f"Person: {self.toa_do_nguoi[0]:.0f},{self.toa_do_nguoi[1]:.0f},{self.toa_do_nguoi[2]:.0f}mm",
                                        (best_x + 20, best_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        else:
                            self.toa_do_nguoi = (0, 0, 0)
                        if detections.class_id is not None and detections.class_id is not None and detections.confidence is not None:
                            color_frame = self.box_annotator.annotate(
                                scene=color_frame,
                                detections=detections,
                                labels=[
                                    f' {self.model.model.names[class_id]} {conf:.2f}'
                                    for class_id, conf in
                                    zip(detections.class_id, detections.confidence)
                                ]
                            )
                        if obstacles.class_id is not None and obstacles.confidence is not None:
                            color_frame = self.box_annotator.annotate(
                                scene=color_frame,
                                detections=obstacles,
                                labels=[
                                    f' {self.model_2.model.names[class_id]} {conf:.2f}'
                                    for class_id, conf in
                                    zip(obstacles.class_id, obstacles.confidence)
                                ]
                            )
                self.latest_display_frame = color_frame
        except Exception as e:
            print("Lỗi trong tracking_people:", e)
    def get_positions(self):
        with self.lock:
            return {
                'nguoi': self.toa_do_nguoi,
                'vat_can': self.toa_do_vat_can
            }

