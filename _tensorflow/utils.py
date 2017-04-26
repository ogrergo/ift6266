import os
import pickle as pkl
import glob
import numpy as np
import PIL.Image as Image

import tensorflow as tf
from urllib.request import urlopen

dataset = "../dataset"

mscoco = os.path.join(dataset, "inpainting")
train_images = os.path.join(mscoco, "train2014")
valid_images = os.path.join(mscoco, "val2014")
caption_path = os.path.join(mscoco, "dict_key_imgID_value_caps_train_and_valid.pkl")

embeddings_folder = os.path.join(dataset, 'embeddings')
embeddings_file = os.path.join(embeddings_folder, "embeddings.npy")
sentence_mapping_file = os.path.join(embeddings_folder, "embeddings_mapping.pkl")

processed_dataset_folder = 'dataset'
processed_dataset_files = os.path.join(processed_dataset_folder, "image_captions_embeddings.*.tfrecords")

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
        with open(sentence_mapping_file, 'rb') as fd:
            sentence_mapping = pkl.load(fd)


def get_embeddings(filelist=None, batch_size=None):
    load_embeddings_data()

    if filelist:
        sentences = []
        for f in filelist:
            keys = caption_dict[os.path.basename(f)[0:-4]]
            np.random.shuffle(keys)
            for k in keys:
                try:
                    s = sentence_mapping[k]
                    break
                except KeyError:
                    continue
            if s == None:
                sentences.append(np.zeros((1024,), dtype=np.float32))
            else:
                sentences.append(embeddings[s])
    else:
        nb_images = embeddings.shape[0]
        sentences = np.array( [embeddings[np.random.randint(nb_images)] for _ in range(batch_size)])
    return sentences


    # return np.zeros((len(filelist) if filelist is not None else batch_size, 1024))

def get_dataset_files(add_valid=False):
    result = glob.glob(os.path.join(train_images, "*.jpg"))

    if add_valid:
        result.extend(glob.glob(os.path.join(valid_images, "*.jpg")))

    return result


def get_processed_dataset_files():
    return glob.glob(processed_dataset_files)

def _preprocess_image(image):
    return (tf.cast(image, tf.float32) / 255.) * 2. - 1.


def _preprocess_embeddings(embeddings, nb_embeddings):

    # random linear combo of embeddings
    combo = tf.random_uniform([nb_embeddings], minval=0, maxval=1)
    combo = combo / tf.reduce_sum(combo)

    k = tf.reduce_sum(tf.transpose(embeddings) * combo, 1)
    # print(k.get_shape())
    return k


def decode_example_record(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'embeddings': tf.FixedLenFeature([], tf.string),
            'embeddings_len': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [64, 64, 3])

    embeddings_len = tf.cast(features['embeddings_len'], tf.int32)

    embeddings = tf.decode_raw(features['embeddings'], tf.float32)
    embeddings = tf.reshape(embeddings, [embeddings_len, 1024])

    return _preprocess_image(image), \
           _preprocess_embeddings(embeddings, embeddings_len), \
           _preprocess_embeddings(embeddings, embeddings_len) # fake captions to be shuffled

def _shuffle_queue(tensor, capacity, min_after_dequeue):
    q = tf.RandomShuffleQueue(capacity, min_after_dequeue, tensor.get_shape())
    tensor

    tf.tuple((q.enqueue_many([tensor]))


def input_pipeline(filenames, batch_size, read_threads, nb_repeat_exemples=1, num_epochs=None, z_dim=100, embedding_size=1024):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    example_list = []
    for _ in range(read_threads):
        img, emb0, emb1 = decode_example_record(filename_queue)
        example_list += (img, emb0, _shuffle_queue(emb1, capacity=capacity))


    image_batch, captions_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return image_batch, captions_batch, \
           tf.random_normal(stddev=0.02, shape=(batch_size, embedding_size)), tf.random_normal(shape=(batch_size, z_dim))

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
