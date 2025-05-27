import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import math
import matplotlib.pyplot as plt

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,30)
pipeline.start(config)
# Lấy profile
profile = pipeline.get_active_profile()
# Lấy thông số nội tại của dòng video màu
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pipeline.stop()

# Khởi tạo bộ phát hiện và theo dõi bàn tay
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
)

# Khởi tạo bộ vẽ
mp_draw = mp.solutions.drawing_utils

class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()

def calculate_angle(v1, v2):
    try:
        # Chuyển đổi tuple thành numpy arrays
        v1 = np.array(v1)
        v2 = np.array(v2)

        # Tính tích vô hướng
        dot_product = np.dot(v1, v2)

        # Tính độ dài của mỗi vectơ
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Tính cos(theta)
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

        # Tính góc (đổi từ radian sang độ)
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)
    except:
        return 0

point = (400, 300)
def show_distance(event, x, y, flags, param):
    global point
    point = (x, y)

# Khởi tạo Camera Intel RealSense
dc = DepthCamera()

# Tạo cửa sổ và thiết lập callback chuột
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)
time_new=0

# plt.ion()
# fig, ax= plt.subplots(2,3)
# angle_cal_a,time_a=[],[]
# angle_cal_b,time_b=[],[]
# angle_cal_c,time_c=[],[]
# angle_cal_d,time_d=[],[]
# angle_cal_e,time_e=[],[]
# line_a, =ax[0,0].plot(time_a,angle_cal_a)
# ax[0,0].set_title('THUMB')
# ax[0,0].set_ylabel('Angle(degree)')
# line_b, =ax[0,1].plot(time_b,angle_cal_b)
# ax[0,1].set_title('INDEX_FINGER')
# ax[0,1].set_ylabel('Angle(degree)')
# line_c, =ax[0,2].plot(time_c,angle_cal_c)
# ax[0,2].set_title('MIDDLE_FINGER')
# ax[0,2].set_ylabel('Angle(degree)')
# line_d, =ax[1,0].plot(time_d,angle_cal_d)
# ax[1,0].set_title('RING_FINGER')
# ax[1,0].set_ylabel('Angle(degree)')
# line_e, =ax[1,1].plot(time_e,angle_cal_e)
# ax[1,1].set_title('PINKY')
# ax[1,1].set_ylabel('Angle(degree)')
#
# ax[0, 0].set_xlim(0, 1)
# ax[0, 0].set_ylim(100, 200)
# ax[0, 1].set_xlim(0, 1)
# ax[0, 1].set_ylim(-5, 200)
# ax[0, 2].set_xlim(0, 1)
# ax[0, 2].set_ylim(-5, 200)
# ax[1, 0].set_xlim(0, 1)
# ax[1, 0].set_ylim(-5, 200)
# ax[1, 1].set_xlim(0, 1)
# ax[1, 1].set_ylim(-5, 200)
# plt.tight_layout()

new_a=0
new_b=0
new_c=0
new_d=0
new_e=0

while True:
    ret, depth_frame, color_frame = dc.get_frame()
    cv2.circle(color_frame, (320, 240), 3, (255, 0, 0), 2)  # Vẽ điểm trung tâm
    cv2.circle(color_frame, point, 2, (0, 255, 0), 1)  # Vẽ landmark
    imgRGB = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if (0 <= point[1] < depth_frame.shape[0]) and (0 <= point[0] < depth_frame.shape[1]):
        distance2 = depth_frame[point[1], point[0]]
        v2 = point[0]
        u2 = point[1]
        Xtemp2 = int(distance2 * float((v2 - 320)) / intr.fx)
        Ytemp2 = int(distance2 * float((u2 - 240)) / intr.fy)
        Ztemp2 = distance2
        cv2.putText(color_frame, "{}mm".format((distance2)), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2)
        cv2.putText(color_frame, f"({Xtemp2},{Ytemp2},{Ztemp2})", (point[0], point[1] + 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(color_frame, handLms, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = color_frame.shape
                cy, cx= int(lm.x * w), int(lm.y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    lmList.append([id, cx, cy])
            list_ =[]
            for n, i in enumerate(lmList):
                # Kiểm tra chỉ số trước khi truy cập depth_frame
                if (0 <= i[2] < depth_frame.shape[0]) and (0 <= i[1] < depth_frame.shape[1]):
                    distance = depth_frame[i[2], i[1]]
                    u = i[2]
                    v = i[1]
                    Xtemp = int(distance * float((u - 320)) / intr.fx)
                    Ytemp = int(distance * float((v - 240)) / intr.fy)
                    Ztemp = distance
                    list_.append((n,Xtemp,Ytemp,int(Ztemp)))
                    # cv2.putText(color_frame, f"{n}", (i[2]+10, i[1]),cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

                finger_tips_A = [4, 8, 12, 16,20]
                vectoBA = []
                vectoBC = []
                for i in finger_tips_A:
                    if len(list_) >= 21:
                        vectoBA.append((list_[i][1] - list_[i-3][1],list_[i][2] - list_[i-3][2],list_[i][3] - list_[i-3][3]))
                for i in finger_tips_A:
                    if len(list_) >= 21:
                        vectoBC.append((list_[0][1] - list_[i][1],list_[0][2] - list_[i][2],list_[0][3] - list_[i][3]))
                angle_cal = []
                angle_cal_real = []
                for i in range(5):
                    if len(vectoBC) >= 5:
                        angle_cal.append(calculate_angle(vectoBA[i], vectoBC[i]))
                    if len(angle_cal)>=5:
                        angle_cal_1=np.interp(angle_cal[0],[150,176],[85,179])
                        angle_cal_2=np.interp(angle_cal[1],[40,170],[0,179])
                        angle_cal_3=np.interp(angle_cal[2],[40,170],[0,178])
                        angle_cal_4=np.interp(angle_cal[3],[30,170],[0,179])
                        angle_cal_5=np.interp(angle_cal[4],[30,165],[0,178])
                        # angle_cal_real.append((float(angle_cal_1),float(angle_cal_2),float(angle_cal_3),float(angle_cal_4),float(angle_cal_5)))
                        np_angle_cal_1 = np.array(angle_cal_1)
                        np_angle_cal_2 = np.array(angle_cal_2)
                        np_angle_cal_3 = np.array(angle_cal_3)
                        np_angle_cal_4 = np.array(angle_cal_4)
                        np_angle_cal_5 = np.array(angle_cal_5)
                        if not np.isnan(np_angle_cal_1):
                            cv2.putText(color_frame, "Thumb:{} degree".format(int(angle_cal_1)), (10, 20),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.5, (0, 255, 0), 2)
                        if not np.isnan(np_angle_cal_2):
                            cv2.putText(color_frame, "Index:{} degree".format(int(angle_cal_2)), (10, 45),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.5, (0, 255, 0), 2)
                        if not np.isnan(np_angle_cal_3):
                            cv2.putText(color_frame, "Middle:{} degree".format(int(angle_cal_3)), (10, 70),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.5, (0, 255, 0), 2)
                        if not np.isnan(np_angle_cal_4):
                            cv2.putText(color_frame, "Ring:{} degree".format(int(angle_cal_4)), (10, 95),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.5, (0, 255, 0), 2)
                        if not np.isnan(np_angle_cal_5):
                            cv2.putText(color_frame, "Pinky:{} degree".format(int(angle_cal_5)), (10, 120),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.5, (0, 255, 0), 2)
                        #Vẽ đồ thị
                        # if angle_cal_1 is not None:
                        #     new_a=new_a+0.005
                        #     time_a.append(new_a)
                        #     angle_cal_a.append(angle_cal_real[0][0])
                        #     if time_a or angle_cal_a:
                        #         line_a.set_xdata(time_a)
                        #         line_a.set_ydata(angle_cal_a)
                        #         ax[0,0].set_xlim(max(time_a)-0.1,max(time_a)+0.1)
                        #         plt.draw()
                        # if angle_cal_2 is not None:
                        #     new_b=new_b+0.005
                        #     time_b.append(new_b)
                        #     angle_cal_b.append(angle_cal_real[0][1])
                        #     if time_b or angle_cal_b:
                        #         line_b.set_xdata(time_b)
                        #         line_b.set_ydata(angle_cal_b)
                        #         ax[0,1].set_xlim(max(time_b)-0.1,max(time_b)+0.1)
                        #         plt.draw()
                        # if angle_cal_3 is not None:
                        #     new_c=new_c+0.005
                        #     time_c.append(new_c)
                        #     angle_cal_c.append(angle_cal_real[0][2])
                        #     if time_c or angle_cal_c:
                        #         line_c.set_xdata(time_c)
                        #         line_c.set_ydata(angle_cal_c)
                        #         ax[0,2].set_xlim(max(time_c)-0.1,max(time_c)+0.1)
                        #         plt.draw()
                        # if angle_cal_4 is not None:
                        #     new_d=new_d+0.005
                        #     time_d.append(new_d)
                        #     angle_cal_d.append(angle_cal_real[0][3])
                        #     if time_d or angle_cal_d:
                        #         line_d.set_xdata(time_d)
                        #         line_d.set_ydata(angle_cal_d)
                        #         ax[1,0].set_xlim(max(time_d)-0.1,max(time_d)+0.1)
                        #         plt.draw()
                        # if angle_cal_5 is not None:
                        #     new_e=new_e+0.005
                        #     time_e.append(new_e)
                        #     angle_cal_e.append(angle_cal_real[0][4])
                        #     if time_e or angle_cal_e:
                        #         line_e.set_xdata(time_e)
                        #         line_e.set_ydata(angle_cal_e)
                        #         ax[1,1].set_xlim(max(time_e)-0.1,max(time_e)+0.1)
                        #         plt.draw()
                        #         plt.pause(0.2)

    # Hiển thị khung hình
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == ord("q"):  # Nhấn 'q' để thoát
        break

dc.release()
cv2.destroyAllWindows()