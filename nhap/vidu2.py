#xác định tọa độ đối tượng
import cv2
#doc anh
cap=cv2.VideoCapture(1)

while True:
    ret,frame=cap.read()
    img = cv2.resize(frame,(0,0),fx=1.5,fy=1.5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img_thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)

    contours,_=cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    draw = cv2.drawContours(img,contours,-1,(0,255,0),1)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] !=0:
            cX = int(M["m10"]/ M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX,cY),1,(0,0,255),1)
            X = str(cX)
            Y = str(cY)
            cv2.putText(img,"("+X+","+Y+")",(cX-30,cY-30),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,0,0),1 )




    cv2.imshow("cua so hien thi",img)
    if cv2.waitKey(1)==ord(" "):
        break

cap.release()
cv2.destroyAllWindows()