import cv2
import matplotlib.pyplot as plt

# Path ke gambar dan file label
image_path = 'data/train/images/2012-09-11_15_16_58_jpg.rf.61d961a86c9a16694403dfcb72cd450c.jpg'
label_path = 'data/train/labels/2012-09-11_15_16_58_jpg.rf.61d961a86c9a16694403dfcb72cd450c.txt'

# Baca gambar
image = cv2.imread(image_path)
h, w, _ = image.shape

# Baca file label YOLO
with open(label_path, 'r') as f:
    lines = f.readlines()

# Gambar bounding box ke gambar
for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * w
    y_center = float(parts[2]) * h
    box_width = float(parts[3]) * w
    box_height = float(parts[4]) * h

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    color = (0, 0, 255) if class_id == 1 else (255, 0, 0) 
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Tampilkan hasil
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Hasil Anotasi Bounding Box")
plt.show()
