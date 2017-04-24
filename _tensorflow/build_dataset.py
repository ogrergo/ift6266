import os
from collections import OrderedDict

import tensorflow as tf
import tqdm
from PIL import Image
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_as_dataset(filelist, caption_dict, embeddings, word_dict, nb_per_file=1024, start_index=0, dataset_dir="dataset"):
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    print(" [*] Creating serialized dataset (%d) in %d examples files."%(len(filelist), nb_per_file))

    def _get_tfrecord_writer(index):
        name = os.path.join(dataset_dir, 'image_captions_embeddings.%d.tfrecords' % index)
        print("\n [*] Writing %s"%name)
        return tf.python_io.TFRecordWriter(name)

    index_file = start_index
    index_example = 0
    writer = _get_tfrecord_writer(index_file)

    np.random.shuffle(filelist)

    for f in tqdm.tqdm(filelist):
        name = os.path.basename(f)[0:-4]

        sentences = OrderedDict(
            {c: embeddings[word_dict[c]] for c in caption_dict[name] if c in word_dict})

        if len(sentences) == 0:
            print("\n [!] the example %s has no embeddings compatible"%f)
            continue

        # captions = [s for s in sentences.keys()]
        captions_embeddings = np.array([c for c in sentences.values()])

        assert captions_embeddings.shape[1] == 1024

        image = (np.array(Image.open(f)).astype(np.float32) / 255.) * 2. - 1.

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image.tostring()),
                    'embeddings_len': _int64_feature(len(sentences)),
                    'embeddings': _bytes_feature(captions_embeddings.tostring())
                }
            )
        )

        serialized = example.SerializeToString()
        writer.write(serialized)
        index_example += 1

        if index_example == nb_per_file:
            index_example = 0
            index_file += 1
            writer.close()
            writer = _get_tfrecord_writer(index_file)

    writer.close()

if __name__ == '__main__':

    import utils as ut
    import os

    print(" [*] Loading embeddings data")
    ut.load_embeddings_data()

    save_as_dataset(ut.get_dataset_files(add_valid=True), ut.caption_dict, ut.embeddings, ut.sentence_mapping)