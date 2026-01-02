from __future__ import annotations
import requests

class BloodCancerAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def predict(self, image_path: str):
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "application/octet-stream")}
            r = requests.post(f"{self.base_url}/predict", files=files, timeout=120)
            r.raise_for_status()
            return r.json()

    def stats(self):
        r = requests.get(f"{self.base_url}/stats", timeout=30)
        r.raise_for_status()
        return r.json()
