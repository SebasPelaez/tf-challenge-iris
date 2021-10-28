from pathlib import Path
import numpy as np

img_path = Path.cwd() 
img_path = img_path / "dataset/"
img_paths = [img.stem for img in img_path.glob("*.json")]
sampled_group = np.random.choice(5, len(img_paths))

with open("img_metadata.csv", "w") as file:
    for img_path, group in zip(img_paths, sampled_group):
        file.write(img_path + "," + str(group) + "\n")
