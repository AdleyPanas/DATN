import socket
import time


HOST = "192.168.0.107"  # Địa chỉ IP của LabVIEW (hoặc dùng "localhost")
PORT = 6000  # Port phải khớp với LabVIEW

# Tạo socket TCP
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Kết nối đến LabVIEW
client.connect((HOST, PORT))
print("Đã kết nối đến LabVIEW!")
dulieu3=0
while True:
    dulieu1=dulieu3
    dulieu2=dulieu1+1
    dulieu3=dulieu2+1
    # Gửi dữ liệu
    data = f"{dulieu1},{dulieu2},{dulieu3},1\r\n"
    client.sendall(data.encode())  # Chuyển chuỗi thành bytes
    data2 = f"{dulieu1},{dulieu2},{dulieu3},0\r\n"
    client.sendall(data2.encode())
    print(f"Đã gửi: {data}")
    time.sleep(0.5)
# Đóng kết nối
client.close()
print("Đã gửi dữ liệu và đóng kết nối.")
