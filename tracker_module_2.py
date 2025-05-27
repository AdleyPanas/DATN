import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import threading
import mediapipe as mp
from frame import *
import time
class PeopleTracker:
    def __init__(self,show_display=True,draw_box=False):
        # Khởi tạo model và các biến cần thiết
        # Model phát hiện người
        self.model = YOLO('model/yolov8n.pt')
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        self.toa_do_nguoi = (0, 0, 0)
        self.toa_do_vat_can = []
        self.running = False
        self.lock = threading.Lock()
        self.dc = DepthCamera()
        self.show_display = show_display
        self.latest_display_frame = None
        self.draw_box = draw_box

        # Khởi tạo face detection của Mediapipe
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
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
        prev_time = time.time()
        try:
            while self.running:
                ret, depth_frame, color_frame, intrinsics = self.dc.get_frame()
                if not ret:
                    continue
                cx, cy = intrinsics.ppx, intrinsics.ppy
                fx, fy = intrinsics.fx, intrinsics.fy
                # Phát hiện người và vật cản
                results = self.model.predict(source=color_frame, stream=False, device="0")[0]
                detections = sv.Detections.from_yolov8(results)
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                print("Tốc độ nhận frame từ camera:", fps)
                with self.lock:
                    # Cập nhật tọa độ vật cản
                    self.toa_do_vat_can = []
                    closest_distance = float('inf')
                    closest_person_coord = (0, 0, 0)
                    for i,box in enumerate(detections.xyxy):
                        class_id = int(detections.class_id[i])
                        conf = detections.confidence[i]
                        x1, y1, x2, y2 = box.astype(int)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if class_id==39:
                            depth = self.get_distance(depth_frame, center_x, center_y)
                            coord = self.tinh_toado(center_x, center_y, depth, cx, cy, fx, fy)
                            self.toa_do_vat_can.append(coord)
                            cv2.putText(color_frame, f"{coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f}mm",
                                        (center_x + 20, center_y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                        elif class_id==0:
                            person_crop = color_frame[y1:y2, x1:x2]
                            results_face = self.face_detection.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                            if results_face.detections:
                                for det in results_face.detections:
                                    bbox = det.location_data.relative_bounding_box
                                    fx1 = int(bbox.xmin * person_crop.shape[1]) + x1
                                    fy1 = int(bbox.ymin * person_crop.shape[0]) + y1
                                    fx2 = fx1 + int(bbox.width * person_crop.shape[1])
                                    fy2 = fy1 + int(bbox.height * person_crop.shape[0])
                                    face_cx = (fx1 + fx2) // 2
                                    face_cy = (fy1 + fy2) // 2
                                    distance = self.get_distance(depth_frame, face_cx, face_cy)

                                    if 0 < distance < closest_distance:
                                        closest_distance = distance
                                        closest_person_coord = self.tinh_toado(face_cx, face_cy, distance, cx, cy, fx,fy)
                                        best_x, best_y = face_cx, face_cy
                    self.toa_do_nguoi = closest_person_coord
                    if closest_distance < float('inf'):
                        cv2.circle(color_frame, (best_x, best_y), 5, (0, 0, 255), -1)
                        cv2.putText(color_frame,
                                    f"Person: {self.toa_do_nguoi[0]:.0f},{self.toa_do_nguoi[1]:.0f},{self.toa_do_nguoi[2]:.0f}mm",
                                    (best_x + 20, best_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    else:
                        self.toa_do_nguoi = (0, 0, 0)
                    if self.draw_box:
                        detect_box = detections[np.isin(detections.class_id, [0, 39])]
                        if detect_box.class_id is not None and detect_box.confidence is not None:
                            color_frame = self.box_annotator.annotate(
                                scene=color_frame,
                                detections=detect_box,
                                labels=[
                                    f' {self.model.model.names[class_id]} {conf:.2f}'
                                    for class_id, conf in
                                    zip(detections.class_id, detections.confidence)
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

