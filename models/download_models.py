"""
Model download utility for PhantomRoad.
Downloads the lightweight MobileNet-SSD Caffe model (~22MB) for OpenCV DNN road scene detection.
"""

import os
import urllib.request
import sys

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
}


def download_models() -> bool:
    """
    Downloads MobileNet-SSD deploy prototxt and caffemodel files into models/.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("PhantomRoad: Downloading MobileNet-SSD for OpenCV DNN...")
    print("=" * 60)

    for filename, url in MODEL_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[FOUND] {filename} already exists ({os.path.getsize(dest) / 1024 / 1024:.2f} MB).")
            continue

        print(f"[DOWNLOADING] {filename} from {url}...")
        try:
            def progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / max(1, total_size))
                sys.stdout.write(f"\r  Progress: {percent}% ({count * block_size / 1024 / 1024:.1f} MB)")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, dest, reporthook=progress)
            print(f"\n[SAVED] {dest} ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"\n[FAILED] Could not download {filename}: {e}")
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass
            return False

    print("\n[SUCCESS] MobileNet-SSD model is ready for local OpenCV DNN inference.")
    return True


if __name__ == "__main__":
    download_models()
