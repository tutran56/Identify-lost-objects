import time
import cv2
import argparse
import numpy as np
from imutils.video import VideoStream
import imutils
import pyglet
import os
import sys

# Cài đặt tham số đọc weight, config và class name
ap = argparse.ArgumentParser(description='Phát hiện đối tượng và cảnh báo khi mất đối tượng.')
ap.add_argument('-o', '--object_name', required=True,
                help='Tên đối tượng cần phát hiện (ví dụ: "cell phone")')
ap.add_argument('-f', '--frame', default=5, type=int,
                help='Số lượng frame không có đối tượng để kích hoạt báo động')
ap.add_argument('-c', '--config', default='yolov3.cfg',
                help='Đường dẫn tới file cấu hình YOLO (.cfg)')
ap.add_argument('-w', '--weights', default='yolov3.weights',
                help='Đường dẫn tới file trọng số YOLO (.weights)')
ap.add_argument('-cl', '--classes', default='yolov3.txt',
                help='Đường dẫn tới file chứa tên các lớp (.txt)')
args = ap.parse_args()

# Kiểm tra sự tồn tại của các file
def check_file(path, description):
    if not os.path.isfile(path):
        print(f"Lỗi: {description} không tồn tại tại đường dẫn: {path}")
        sys.exit(1)

check_file(args.config, "File cấu hình (.cfg)")
check_file(args.weights, "File trọng số (.weights)")
check_file(args.classes, "File danh sách lớp (.txt)")

# Hàm trả về output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        # Các phiên bản OpenCV khác nhau có thể có định dạng khác nhau cho getUnconnectedOutLayers
        unconnected_out_layers = net.getUnconnectedOutLayers()
        if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 2:
            unconnected_out_layers = unconnected_out_layers.flatten()
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        return output_layers
    except Exception as e:
        print(f"Lỗi khi lấy output layers: {e}")
        sys.exit(1)

# Hàm vẽ các hình chữ nhật và tên lớp
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Đọc từ webcam
print("[INFO] Bắt đầu mở webcam...")
cap = VideoStream(src=0).start()
time.sleep(2.0)  # Đợi webcam khởi động

# Đọc tên các lớp
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Tạo màu ngẫu nhiên cho các lớp
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Đọc mạng YOLO
try:
    print("[INFO] Đang tải mô hình YOLO...")
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] Mô hình YOLO đã được tải thành công.")
except cv2.error as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    sys.exit(1)

nCount = 0

# Bắt đầu đọc từ webcam
print("[INFO] Bắt đầu phát hiện đối tượng...")
while True:
    # Đọc frame
    frame = cap.read()
    if frame is None:
        print("Không thể đọc frame từ webcam.")
        break
    image = imutils.resize(frame, width=600)

    # Biến theo dõi đối tượng có tồn tại trong khung hình hay không
    isExist = False

    # Resize và đưa khung hình vào mạng để dự đoán
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    # Lọc các đối tượng trong khung hình
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > conf_threshold) and (classes[class_id].lower() == args.object_name.lower()):
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Áp dụng Non-Maximum Suppression để loại bỏ các bounding box trùng lặp
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Vẽ các bounding box quanh đối tượng
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if classes[class_ids[i]].lower() == args.object_name.lower():
            isExist = True
            draw_prediction(image, class_ids[i], x, y, x + w, y + h)

    # Nếu tồn tại đối tượng thì reset nCount, ngược lại tăng và kiểm tra
    if isExist:
        nCount = 0
    else:
        nCount += 1
        # Nếu qua quá số frame không có đối tượng, báo động
        if nCount > args.frame:
            cv2.putText(image, "Alarm! Alarm! Alarm!", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            try:
                # Phát âm thanh cảnh báo
                music = pyglet.media.load('police.wav', streaming=False)
                music.play()
            except Exception as e:
                print(f"Lỗi khi phát âm thanh: {e}")

    # Hiển thị kết quả
    cv2.imshow("Object Detection", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Đã dừng phát hiện đối tượng.")
        break

# Giải phóng tài nguyên
cap.stop()
cv2.destroyAllWindows()
    #python3 anti_thief.py -o "cell phone" -f 5 --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
