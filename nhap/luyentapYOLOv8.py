import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv

#khoi tao mo hinh YOLO v8
model = YOLO("yolov8n-face.pt","v8")
#khởi tạo bộ vẽ hộp bao quanh đối tượng
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
#đọc camera
cap = cv2.VideoCapture(1)
#nhận diện và theo dõi đối tượng trong ảnh
while True:
    ret,frame = cap.read()
    if not ret:
        break
    results = model.track(source=frame, show=False, tracker='bytetrack.yaml', stream=True, device= "0")
    for result in results:
        #frame = result.orig_img #lấy ảnh gốc
        detections = sv.Detections.from_yolov8(result)
        #kiem tra neu tra ve ID cua doi tuong, gan ID cho doi tuong
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        #chỉ lấy đối tượng là con người (ID=0), bỏ các đối tượng khác
        detections = detections[detections.class_id == 0]
        #kiểm tra nếu không có đối tượng thì hiển thị ảnh gốc
        print(detections.xyxy.size)
        if detections.xyxy.size == 0 or (detections.tracker_id is None or not detections.tracker_id.any()):
            cv2.imshow("camera", frame)
        else:
            #nếu có đối tượng thì vẽ box lên hình
            labels = [
                f'id: {tr_id} {model.model.names[int(cls_id)]} {conf:.2f}'
                for tr_id, cls_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            # lấy đối tượng có ID = 1
            xyxy = detections.xyxy[detections.tracker_id == 1]
            # xác định tâm đối tượng và vẽ tâm
            if np.array(xyxy).size != 0:

                x1,y1,x2,y2 = np.array(xyxy.astype(int))[0]
                x=(x1+x2)//2
                y=(y1+y2)//2
                cv2.circle(frame,(x,y),5,(0,0,255),-1)

            cv2.imshow("camera",frame)
    if cv2.waitKey(1) == ord(" "):
        break
cap.release()
cv2.destroyAllWindows()