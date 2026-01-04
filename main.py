import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub

# MoveNet ve SSD MobilenetV2 modellerini TensorFlow Hub'dan yükle
pose_model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
yolo_model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

# Web kamerasını başlat
cap = cv2.VideoCapture(0)


# Poz anahtar noktalarını almak için fonksiyon
def get_pose_keypoints(image):
    input_image = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Model tahminleri
    outputs = pose_model.signatures['serving_default'](input_image)
    keypoints = outputs['output_0'].numpy()
    keypoints = keypoints.reshape((17, 3))  # 17 anahtar nokta ve her biri (x, y, güven) formatında

    return keypoints


# Anahtar noktaları görüntü üzerine çizmek için fonksiyon
def draw_keypoints(image, keypoints):
    for x, y, confidence in keypoints:
        if confidence > 0.5:  # Güven eşiği
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)


# YOLO tahminlerini almak için fonksiyon
def get_yolo_predictions(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]

    result = yolo_model(img)

    return result


# YOLO bounding box'ları görüntü üzerine çizmek için fonksiyon
def draw_yolo_boxes(image, yolo_predictions):
    for i in range(int(yolo_predictions['num_detections'])):
        detection_box = yolo_predictions['detection_boxes'][0][i].numpy()
        ymin, xmin, ymax, xmax = detection_box
        (left, right, top, bottom) = (
        xmin * image.shape[1], xmax * image.shape[1], ymin * image.shape[0], ymax * image.shape[0])
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Poz anahtar noktalarını al ve çerçeve üzerine çiz
    keypoints = get_pose_keypoints(frame)
    draw_keypoints(frame, keypoints)

    # YOLO tahminlerini al ve bounding box'ları çiz
    yolo_predictions = get_yolo_predictions(frame)
    draw_yolo_boxes(frame, yolo_predictions)

    # Sonucu göster
    cv2.imshow('Pose and YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
