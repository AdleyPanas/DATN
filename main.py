from tracker_module_2 import PeopleTracker
from map_animation import RealTimeVisualizer
import matplotlib.pyplot as plt
from Astar import *
import time
import threading
import cv2
import numpy as np
positions_shared = {
    "nguoi": None,
    "vat_can": []
}
lock = threading.Lock()
shared_display_frame = {"frame": None}

def toa_do_map(x, y,size=100):
    def convert(coord):
        return ((coord + size/2) // size) * size
    xm = int(convert(x))
    ym = int(convert(y))
    return xm, ym
def tracking_thread_func(tracker,shared_frame):
    while True:
        positions = tracker.get_positions()
        with lock:
            positions_shared["nguoi"] = positions["nguoi"]
            positions_shared["vat_can"] = positions["vat_can"]
        with tracker.lock:
            if tracker.latest_display_frame is not None:
                shared_frame["frame"] = tracker.latest_display_frame.copy()  # <-- Gán frame để hiển thị
        time.sleep(0.01)
def map_thread_func(visualizer):
    while True:
        visualizer.update_data_from_thread()
        time.sleep(0.1)
def main():
    tracker = PeopleTracker(show_display=True)
    tracker.start_tracking()
    pf = PathFinder(grid_width=3800, grid_depth=3000, grid_size=100)
    visualizer = RealTimeVisualizer(pf)  # khởi tạo 1 lần
    visualizer.start_animation()
    # Thread nhận dạng
    t1 = threading.Thread(target=tracking_thread_func,args=(tracker,shared_display_frame), daemon=True)
    t1.start()
    t2 = threading.Thread(target=map_thread_func, args=(visualizer,), daemon=True)
    t2.start()
    last_astar_time = 0   #Biến này để kiểm soát tần suất chạy A*
    astar_interval = 1   #Chạy A* mỗi 1 giây
    prev_time = time.time()
    try:
        while True:
            with lock:
                frame = shared_display_frame["frame"]
                nguoi = positions_shared["nguoi"]
                vat_can = positions_shared["vat_can"]
                nguoi = np.array(nguoi).astype(int)
                vat_can = np.array(vat_can).astype(int) if len(vat_can) > 0 else np.array([])
            if vat_can.size > 0:
                obstacle_coords = [toa_do_map(x, z, 100) for (x, _, z) in vat_can]
            else:
                obstacle_coords =[]
            pf.mark_obstacles_from_camera(obstacle_coords)
            visualizer.obstacle_coords = obstacle_coords
            if frame is not None:
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                print("Tốc độ nhận frame từ Thread:", fps)
                cv2.imshow("Tracking View", frame)
                if cv2.waitKey(1) & 0xFF == ord(" "):
                    tracker.stop_tracking()
                    break
            if len(nguoi)>0:
                now = time.time()
                if now - last_astar_time >= astar_interval:
                    pf.a_star(start_coord=(0, 0), end_coord=toa_do_map(nguoi[0], nguoi[2], 100))
                    last_astar_time = now
            plt.pause(0.001)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Đang dừng chương trình...")
        tracker.stop_tracking()



if __name__ == "__main__":
    main()