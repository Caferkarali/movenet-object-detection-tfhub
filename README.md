# 👁️ Real-Time Pose Estimation & Object Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/Library-TensorFlow-orange)
![Computer Vision](https://img.shields.io/badge/Topic-Computer%20Vision-blue)

## 📌 Project Overview
This project demonstrates a multi-task computer vision pipeline that performs **Pose Estimation** and **Object Detection** simultaneously on a live webcam feed.

It leverages the power of **TensorFlow Hub** to load pre-trained state-of-the-art models without the need for complex local training. The application is designed to identify human body keypoints while concurrently detecting and bounding objects in the scene.

## 🧠 Models Used

The project integrates two distinct models from TensorFlow Hub:

1.  **Pose Estimation:** [MoveNet (Thunder)](https://tfhub.dev/google/movenet/singlepose/thunder/4)
    * A high-accuracy model designed for detecting 17 human body keypoints.
    * Optimized for fitness, sports, and health applications.

2.  **Object Detection:** [SSD MobileNet V2](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)
    * A Single Shot Multibox Detector (SSD) architecture.
    * Optimized for speed and efficiency on mobile/edge devices.

## 🚀 Features
* **Real-Time Inference:** Processes video frames instantly from the webcam.
* **Keypoint Visualization:** Draws skeleton joints (shoulders, elbows, knees, etc.) on the detected person.
* **Bounding Boxes:** Draws rectangles around detected objects (e.g., person, chair, bottle).
* **TF Hub Integration:** Seamless model loading directly from the cloud.

## 🛠 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/movenet-object-detection-tfhub.git
    cd movenet-object-detection-tfhub
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python main.py
    ```
    *Note: The first run might take a few moments to download the models from TensorFlow Hub.*

## 💻 Code Explanation

The pipeline follows these steps:
1.  **Capture:** OpenCV grabs a frame from the webcam.
2.  **Preprocessing:**
    * For MoveNet: Resizes image to 256x256 and casts to `int32`.
    * For SSD MobileNet: Converts BGR to RGB and casts to `uint8`.
3.  **Inference:** Both models run prediction on the processed frame.
4.  **Visualization:**
    * `draw_keypoints`: Plots high-confidence pose landmarks.
    * `draw_yolo_boxes`: Plots detection bounding boxes (using SSD MobileNet output).
5.  **Display:** The annotated frame is shown to the user.

## 🤝 Contribution
Contributions are welcome!

## 📝 License
This project is open-source and available under the MIT License.
