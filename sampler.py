from itertools import groupby

import scipy.misc
from PIL import Image

from utils import get_border, get_embedding, save_batch
from model import FLAGS, Model
import tensorflow as tf
import pprint
import os
import numpy as np

from utils import get_exemple_from_filelist, get_valid_dataset_filelist

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
      os.makedirs(FLAGS.sample_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
      dcgan = Model(sess)

      if not dcgan.load(FLAGS.checkpoint_dir):
          raise Exception("[!] Train a model first, then run test mode")

      valid_set = get_exemple_from_filelist(get_valid_dataset_filelist())

      random_samples(dcgan, valid_set)
      noise_samples(dcgan, valid_set)
      wrong_captions_samples(dcgan, valid_set)
      caption_variation(dcgan, valid_set)

def random_samples(dcgan, exemple_it):
    g = _get_batch(exemple_it)

    border = get_border(g[0])
    embs = get_embedding(g[1])

    result = dcgan.eval(border=border, caption=embs, z=np.random.normal(size=(64, 100)))
    save_batch(result[0], 'samples', img_tag='random')


def noise_samples(dcgan, exemple_it):
    k = next(exemple_it)
    border = get_border(k[0])
    caption_emb = get_embedding(k[1])

    # make the batch
    batch_border = _make_batch(border)
    batch_captions = _make_batch(caption_emb)

    result = dcgan.eval(border=batch_border, caption=batch_captions, z=np.random.normal(size=(64, 100)))
    save_batch(result[0], 'samples', img_tag='noises')


def wrong_captions_samples(dcgan, exemple_it):
    k0 = _get_batch(exemple_it)
    k1 = _get_batch(exemple_it)
    batch_border = get_border(k0[0])
    batch_captions = get_embedding(k1[1])
    batch_z = _make_batch(np.random.normal(size=(100,)))

    result = dcgan.eval(border=batch_border, caption=batch_captions, z=batch_z)
    save_batch(result[0], 'samples', img_tag='wrong_captions')


def caption_variation(dcgan, exemple_it):
    k0 = _get_batch(exemple_it)
    k1 = next(exemple_it)
    batch_border = _make_batch(get_border(k1[0]))
    batch_captions = get_embedding(k0[1])
    batch_z = _make_batch(np.random.normal(size=(100,)))

    result = dcgan.eval(border=batch_border, caption=batch_captions, z=batch_z)
    save_batch(result[0], 'samples', img_tag='caption_variation')


def _make_batch(val):
    return np.repeat(np.expand_dims(val, 0), 64, axis=0)

def _get_batch(exemple_it, size=64):
    g = next(groupby(enumerate(exemple_it), key=lambda e: e[0] // size))

    g = tuple(zip(*g[1]))  # remove index of enumerate
    g = tuple(zip(*g[1]))  # two list, one img and other is emb
    return (np.array(g[0]), np.array(g[1]))


if __name__ == '__main__':
  tf.app.run()
