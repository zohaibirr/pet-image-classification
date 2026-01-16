import os
import shutil
import random

def split_data(raw_dir, output_dir, split=(0.7, 0.15, 0.15)):
    classes = os.listdir(raw_dir)

    for cls in classes:
        images = os.listdir(os.path.join(raw_dir, cls))
        random.shuffle(images)

        train_end = int(split[0] * len(images))
        val_end = train_end + int(split[1] * len(images))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_imgs in splits.items():
            split_path = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_path, exist_ok=True)

            for img in split_imgs:
                shutil.copy(
                    os.path.join(raw_dir, cls, img),
                    os.path.join(split_path, img)
                )

if __name__ == "__main__":
    split_data("data/raw", "data")
