from glob import glob
import os
import numpy as np
import PIL.Image as Image

data = glob(os.path.join("dataset", "inpainting/train2014", "*.jpg"))
print("check dataset [%d jpg]"%len(data))
for d in data:
    img = np.array(Image.open(d))
    if img.shape != (64, 64, 3):
        print(img.shape)
        img = np.repeat(img[:, :, None], axis=2, repeats=3)
        Image.fromarray(img).save(d)


all(len(d.shape) == 3 for d in data)
