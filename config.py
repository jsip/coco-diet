import os

OUTPUT_DIR = os.path.join("images", "cam")
IP_ADDRESS = "192.168.2.15"
PORT = 5000
HEALTH_ENDPOINT = f"http://{IP_ADDRESS}:{PORT}/"
PREDICT_ENDPOINT = f"http://{IP_ADDRESS}:{PORT}/predict"