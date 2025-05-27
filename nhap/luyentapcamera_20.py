import cv2
import numpy as np
import pyrealsense2 as rs
#import mediapipe as mp
import math
#import matplotlib.pyplot as plt
from frame import *

point = (400,300)

def show_distance(event, x, y, flags, param):
    global point
    point = (x,y)

def get_distance(map,x,y):
    try:
        return map[y,x]
    except:
        print("x={} y= {}".format(x,y))
        return map[y,x]

dc = DepthCamera()

#tao vi tri chuot khi di chuyen chuot
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame",show_distance)
goc=math.atan(150/500)
z_trans=30*math.sin(goc)
y_trans=30-30*math.cos(goc)
#vonglapvideo
while True:
    ret, depth_frame, color_frame, intrinsics = dc.get_frame()
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    fx = intrinsics.fx
    fy = intrinsics.fy
    tam = (int(cx), int(cy))
    #hien thi khoang cach cho mot diem cu the
    x=point[0]
    y=point[1]
    cv2.circle(color_frame,point,4,(0,0,255))
    cv2.circle(color_frame, tam, 4, (0, 0, 255))
    distance = get_distance(depth_frame,point[0],point[1])
    X_cam = round(float(distance * (x - cx) / fx), 0)
    Y_cam = round(float(distance * (y - cy) / fy), 0)
    Z_cam = round(float(distance), 0)
    #đổi tọa độ
    Xs=X_cam
    Ys=Y_cam*math.cos(goc)-Z_cam*math.sin(goc)+y_trans
    Zs = Y_cam*math.sin(goc) + Z_cam*math.cos(goc) - z_trans

    #vẽ tâm
    dis_tam = get_distance(depth_frame,tam[0], tam[1])
    Zt = round(float(dis_tam), 0)
    toado_tam=(0,-Zt*math.sin(goc)+y_trans,Zt*math.cos(goc)-z_trans)
    toa_do = (Xs, Ys, Zs)
    cv2.putText(color_frame, f"{toa_do[0]:,.0f},{toa_do[1]:,.0f},{toa_do[2]:,.0f}", (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0), 2)
    cv2.putText(color_frame, f"{toado_tam[0]:,.0f},{toado_tam[1]:,.0f},{toado_tam[2]:,.0f}", (tam[0] + 10, tam[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0), 2)
    cv2.line(color_frame, (tam[0]-200,tam[1]), (tam[0]+200,tam[1]), (255,0,0), 2)
    cv2.line(color_frame, (tam[0], tam[1]-200), (tam[0], tam[1]+200), (255, 0, 0), 2)

    cv2.imshow("Color frame",color_frame)
    key = cv2.waitKey(1)
    if key == ord(" "):
        break
