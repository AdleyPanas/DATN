import cv2

# Đọc ảnh
image = cv2.imread('images/yolo.jpg')

# Resize ảnh về kích thước (width=400, height=300)
resized_image = cv2.resize(image, (0, 0),fx=0.1,fy=0.1)

# Lưu ảnh đã resize
cv2.imwrite('images/yolo1.jpg', resized_image)

print("Ảnh đã được resize và lưu thành công!")
