import numpy as np
import cv2
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    ret, frame = cap.read()
    #width = int(cap.get(3))
    #height = int(cap.get(4))
    #small_frame = cv2.resize(frame,(0,0),fx=1,fy=1)
    #image=np.zeros((height,width*2,3),np.uint8)

    #image[:height,:width]=frame
    #image[:height,width:]=cv2.rotate(frame,cv2.ROTATE_180)
    #print(ret)
    cv2.imshow("cua so camera2", frame)

    if cv2.waitKey(1)==ord(" "):
        break
cap.release()
cv2.destroyAllWindows()
