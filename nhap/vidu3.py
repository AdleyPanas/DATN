# xác định tọa độ và giá trị điểm màu click chuột trái/phải
import cv2

def event(event,x,y,flag,para):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(img,str(x)+','+str(y),(x,y),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)
        cv2.imshow("cua so hien thi", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        b = img[y,x,0]
        g = img[y,x,1]
        r = img[y,x,2]
        cv2.putText(img,str(b)+','+str(g)+','+str(r),(x,y),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
        cv2.imshow("cua so hien thi", img)

cap = cv2.VideoCapture(1)

while True:
    ret,frame = cap.read()
    img = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    cv2.imshow("cua so hien thi", img)

    cv2.setMouseCallback('cua so hien thi',event)

    if cv2.waitKey(1) == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()