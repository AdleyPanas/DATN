
import socket
import time
import threading


# #khởi tạo truyền thông
def start_tcp_server(shared_data):
    def trans_coord(x,z,xs,zs):
        x_n=x-xs
        z_n=z-zs
        return x_n,z_n
    def tcp_communication(conn, addr):
        print(f"[TCP] Đã kết nối từ {addr}")
        last_astar_time = 0  # Biến này để kiểm soát tần suất gửi vị trí đặt
        astar_interval = 0.1
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                message = data.decode().strip()
                #print(f"LabVIEW gửi: {message}")
                if message == "PING":
                    now = time.time()
                    if now - last_astar_time >= astar_interval:
                        if shared_data["toa_do_dat"]:
                            dat = shared_data["toa_do_dat"].pop(0)
                        last_astar_time = now
                    ph = shared_data.get("toa_do_phan_hoi", (0, 0))
                    dat_n = trans_coord(dat[0],dat[1],ph[0],ph[1])
                    ph_n = trans_coord(0,0,ph[0],ph[1])
                    timestamp = time.time()
                    response = f"{dat_n[0]},{dat_n[1]},{ph_n[0]},{ph_n[1]},{timestamp:.0f}\r\n"
                    print("Đã gửi dữ liệu!",dat,"path còn lại: ",shared_data["toa_do_dat"])
                    conn.sendall(response.encode())
        except Exception as e:
            print(f"[TCP] Lỗi với {addr}: {e}")
        finally:
            conn.close()
            print(f"[TCP] Đã đóng kết nối với {addr}")
    def server_thread():
        host = '0.0.0.0'  # Lắng nghe mọi IP
        port = 12345
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((host,port))
            server.listen(1)
            print("Python server đang chờ LabVIEW kết nối...")
            while True:
                conn, addr = server.accept()
                tcp_thread = threading.Thread(target=tcp_communication, args=(conn, addr), daemon=True)
                tcp_thread.start()

    threading.Thread(target=server_thread, daemon=True).start()
# if __name__ == "__main__":
#     tcp_server = threading.Thread(target=tcp_server_thread, daemon=True)
#     tcp_server.start()
#
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         print("\n[Main] Nhận tín hiệu dừng, thoát chương trình.")
