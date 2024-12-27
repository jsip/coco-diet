import cv2
import threading
import queue
import time
import os
import requests

CAMERA_INDEX = 0
DESIRED_FPS = 10
SCREENSHOT_INTERVAL = 1.0
FRAME_DURATION = 1.0 / DESIRED_FPS

OUTPUT_DIR = os.path.join("images", "cam")
PREDICT_ENDPOINT = "http://localhost:5000/predict"


def ensure_output_dir_exists():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_inference_docker(image_path):
    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        try:
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"\nInterference for {image_path}:")
                print(f"\nPredicted class: {data.get('pred_class')}")
                print(f"\nConfidence: {data.get('confidence'):.4f}")

                probabilities = data.get("probabilities", {})
                print("Class probabilities:")
                for cls_name, prob_val in probabilities.items():
                    print(f"{cls_name}: {prob_val:.4f}")

                print(f"Inference time: {data.get('inference_time')}s\n")

            else:
                print("Error response from model: ", response.text)

        except requests.exceptions.RequestException as e:
            print(f"Error calling Docker model service: {e}")


def screenshot_worker(screenshot_queue):
    while True:
        idx, frame = screenshot_queue.get()
        if idx is None:
            break

        ensure_output_dir_exists()

        filename = os.path.join(OUTPUT_DIR, f"img_{idx}.jpeg")
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")

        run_inference_docker(filename)

        screenshot_queue.task_done()


def start_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    screenshot_queue = queue.Queue()
    worker_thread = threading.Thread(
        target=screenshot_worker,
        args=(screenshot_queue,),
        daemon=True
    )
    worker_thread.start()

    last_screenshot_time = time.time()
    screenshot_count = 0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            if current_time - last_screenshot_time >= SCREENSHOT_INTERVAL:
                screenshot_count += 1
                screenshot_queue.put((screenshot_count, frame))
                last_screenshot_time = current_time

            cv2.imshow('Webcam (10 FPS, screenshot every 1s)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - loop_start
            if elapsed < FRAME_DURATION:
                time.sleep(FRAME_DURATION - elapsed)

    finally:
        screenshot_queue.put((None, None))  # send sentinel
        cap.release()
        cv2.destroyAllWindows()


def main():
    start_camera()


if __name__ == "__main__":
    main()
