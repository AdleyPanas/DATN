import socket

HOST = '0.0.0.0'  # Lắng nghe mọi IP
PORT = 12345      # Cổng kết nối

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print("Python server đang chờ LabVIEW kết nối...")
    ##addr là địa chỉ IP của labview
    conn, addr = s.accept()
    with conn:
        print(f"Đã kết nối từ {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            message = data.decode().strip()
            print(f"LabVIEW gửi: {message}")
            if message == "PING":
                data = f"{toa_do[0]},{toa_do[0]},{toa_do[0]},1\r\n"
                conn.sendall(response.encode())
