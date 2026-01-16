import os
from PIL import Image

def clean_directory(base_dir):
    removed = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                with Image.open(file_path) as img:
                    img.verify()   # check if image is valid
            except Exception:
                print("❌ Removing corrupted file:", file_path)
                os.remove(file_path)
                removed += 1

    print(f"\n✅ Cleaning completed. Removed {removed} corrupted files.")

if __name__ == "__main__":
    clean_directory("data")
