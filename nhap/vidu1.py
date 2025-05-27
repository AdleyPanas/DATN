import cv2
#doc anh
cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    cv2.imshow("camera",frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(width,height)
    if cv2.waitKey(1)==ord(" "):
        break


