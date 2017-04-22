import os
import pickle as pkl
import glob
import numpy as np
import PIL.Image as Image
import pickle

from urllib.request import urlopen

dataset = "../dataset"

mscoco = os.path.join(dataset, "inpainting")
train_images = os.path.join(mscoco, "train2014")
valid_images = os.path.join(mscoco, "val2014")
caption_path = os.path.join(mscoco, "dict_key_imgID_value_caps_train_and_valid.pkl")

embeddings_folder = os.path.join(dataset, 'embeddings')
embeddings_file = os.path.join(embeddings_folder, "embeddings.npy")
sentence_mapping_file = os.path.join(embeddings_folder, "embeddings_mapping.pkl")

def download_embeddings():

    try:
        os.makedirs(embeddings_folder)
    except:
        pass

    file = os.path.join(dataset, 'MSCOCO_captions_embeddings.zip')
    if not os.path.isfile(file):

        embedings_url = 'https://s3.amazonaws.com/akiaj36ibbq3myh2imfa-dump/MSCOCO_captions_embeddings.zip'
        print(" [*] Downloading %s" % embedings_url)

        zip_archive = urlopen(embedings_url)
        with open(file, 'wb') as output:
            output.write(zip_archive.read())


    if not os.path.isfile(embeddings_file) or not os.path.isfile(sentence_mapping_file):
        print(" [*] Extracting files from %s ..." % file)
        import zipfile
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(embeddings_folder)

        os.rename(os.path.join(embeddings_folder, 'out_file.npy'), embeddings_file)
        os.rename(os.path.join(embeddings_folder, 'dico.pkl'), sentence_mapping_file)



def load_images(filelist, middle=True):
    sample = [np.array(Image.open(sample_file)) for sample_file in filelist]
    batch = np.array(sample).astype(np.float32)
    if middle:
        m = batch.shape[1] // 2
        batch = batch[:, m-16:m+16, m-16:m+16, :]
        assert (batch.shape[1] == 32 and batch.shape[2] == 32)
        # Image.fromarray(batch[0,:,:].astype(np.uint8)).show()

    return (2.* batch)/255. - 1.

caption_dict = None
embeddings = None
sentence_mapping = None

def load_embeddings_data():
    global caption_dict, embeddings, sentence_mapping

    if caption_dict is None:
        with open(caption_path, 'rb') as fd:
            caption_dict = pkl.load(fd)

    if embeddings is None:
        embeddings = np.load(embeddings_file)

    if sentence_mapping is None:
        sentence_mapping = pickle.load(sentence_mapping_file)


def get_embeddings(filelist=None, batch_size=None):
    load_embeddings_data()

    if filelist:
        sentences = [embeddings[sentence_mapping[np.random.choice(caption_dict[f])]] for f in filelist]
    else:
        nb_images = embeddings.shape()[0]
        sentences = [embeddings[np.random.randint(nb_images),0] for _ in range(batch_size)]
    return sentences


    # return np.zeros((len(filelist) if filelist is not None else batch_size, 1024))

def get_dataset_files():
    return glob.glob(os.path.join(train_images, "*.jpg"))


# def load_data(start, n):
#     with open(caption_path, 'rb') as fd:
#         caption_dict = pkl.load(fd)
#         pass
#
#     print(data_path + "/*.jpg")
#     imgs = glob.glob(data_path + "/*.jpg")
#
#     imgs = imgs[start: start + n]
#     data = np.empty((n, 3, 64, 64), dtype='uint8')
#
#     for i, img_path in enumerate(imgs):
#         __t = np.array(Image.open(img_path))
#         if len(__t.shape) == 2:
#             __t = np.repeat(__t[None, :, :], axis=0, repeats=3)
#         else:
#             __t = __t.transpose((2,0,1))
#         data[i, :, :, :] = __t
#
#     return data