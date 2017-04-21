import os
import pickle as pkl
import glob
import numpy as np
import PIL.Image as Image

mscoco = "../dataset/inpainting"
split = "train2014"
caption_path = "dict_key_imgID_value_caps_train_and_valid.pkl"

data_path = os.path.join(mscoco, split)
caption_path = os.path.join(mscoco, caption_path)


def load_images(filelist):
    sample = [np.array(Image.open(sample_file)) for sample_file in filelist]
    return np.array(sample).astype(np.float32)

def get_dataset_files():
    return  glob.glob(data_path + "/*.jpg")


def load_data(start, n):
    with open(caption_path, 'rb') as fd:
        caption_dict = pkl.load(fd)

    print(data_path + "/*.jpg")
    imgs = glob.glob(data_path + "/*.jpg")

    imgs = imgs[start: start + n]
    data = np.empty((n, 3, 64, 64), dtype='uint8')

    for i, img_path in enumerate(imgs):
        __t = np.array(Image.open(img_path))
        if len(__t.shape) == 2:
            __t = np.repeat(__t[None, :, :], axis=0, repeats=3)
        else:
            __t = __t.transpose((2,0,1))
        data[i, :, :, :] = __t

    return data