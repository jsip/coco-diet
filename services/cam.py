import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.screenshots import check_model_health, start_camera

CAMERA_INDEX = 0
DESIRED_FPS = 10
SCREENSHOT_INTERVAL = 1.0
FRAME_DURATION = 1.0 / DESIRED_FPS

def main():
    check_model_health()
    start_camera(CAMERA_INDEX, DESIRED_FPS, SCREENSHOT_INTERVAL, FRAME_DURATION)

if __name__ == "__main__":
    main()