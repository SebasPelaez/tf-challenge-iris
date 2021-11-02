from pathlib import Path
import numpy as np

img_path = Path.cwd()
img_path = img_path / "dataset/"
file_name_metadata = "img_metadata.csv"


def list_file_names_sampled_group(img_path, file_name_metadata):
    img_path = Path(img_path)
    img_paths = [img.stem for img in img_path.glob("*.json")]
    sampled_group = np.random.choice(5, len(img_paths))

    with open(file_name_metadata, "w") as file:
        for img_path, group in zip(img_paths, sampled_group):
            file.write(img_path + "," + str(group) + "\n")

def list_file_names(img_path, file_name_metadata):
    img_path = Path(img_path)
    img_paths = [img.stem for img in img_path.glob("*.png")]

    with open(file_name_metadata, "w") as file:
        for img_path in img_paths:
            file.write(img_path + "\n")

if __name__ == "__main__":
    list_file_names_sampled_group(img_path="dataset/train", file_name_metadata="img_metadata_train_dev.csv")
    list_file_names(img_path="dataset/test", file_name_metadata="img_metadata_test.csv")