'''
version 7:
- YOLO human tracking
'''

from ultralytics import YOLO
import numpy as np
import supervision as sv
import cv2
cap = cv2.VideoCapture(0)
# Khởi tạo mô hình YOLOv8
model = YOLO('yolov8m.pt', 'v8')
# Khởi tạo bộ vẽ hộp bao quanh đối tượng
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
# Nhận diện & theo dõi đối tượng trong ảnh

while True:
    ret,image = cap.read()
    results = model.track(source=image, show=False, tracker='bytetrack.yaml', stream=True, device= "0")

    for result in results:
        frame = result.orig_img

        detections = sv.Detections.from_yolov8(result)
        # kiem tra neu tra ve ID cua doi tuong
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[detections.class_id == 0]

        # import IPython; IPython.embed()

        # Check if detections exist
        if detections.xyxy.size == 0:
            # No detections found, simply show the original frame without annotations
            cv2.imshow('Detection Results', frame)
        else:
            # Detections found, proceed with annotating and showing the frame
            labels = [
                f'id: {tr_id} {model.model.names[int(cls_id)]} {conf:.2f}'
                for tr_id, cls_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Show the annotated frame using OpenCV
            cv2.imshow('Detection Results', frame)

        # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video writer and close OpenCV windows
# video_writer.release()
cap.release()
cv2.destroyAllWindows()

