from glob import glob
import os

import itertools as it
import numpy as np
import PIL.Image as Image

#data = glob(os.path.join("dataset", "inpainting/train2014", "*.jpg"))
#print("check dataset [%d jpg]"%len(data))
#for d in data:
 #   img = np.array(Image.open(d))
  #  if img.shape != (64, 64, 3):
   #     print(img.shape)
    #    img = np.repeat(img[:, :, None], axis=2, repeats=3)
     #   Image.fromarray(img).save(d)

caption_path = "dict_key_imgID_value_caps_train_and_valid.pkl"
import pickle as pkl

with open('dataset/inpainting/' + caption_path, 'rb') as fd:
    caption_dict = pkl.load(fd)

with open("dataset/sentence_list.txt", 'w') as fp:
    fp.writelines(['%s\n'%s for s in it.chain.from_iterable(caption_dict.values())])



