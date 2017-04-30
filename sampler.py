from itertools import groupby

from utils import get_border, get_embedding, save_batch, to_float, save_image, interpolate_noise, _make_batch, \
    _get_batch
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

      # caption_random_combo(dcgan, valid_set)
      # random_samples(dcgan, valid_set)
      noise_samples(dcgan, valid_set)
      # wrong_captions_samples(dcgan, valid_set)
      # caption_variation(dcgan, valid_set)


def random_samples(dcgan, exemple_it):
    g = _get_batch(exemple_it)

    border = to_float(g[0])
    embs = get_embedding(g[1])

    result = dcgan.eval(image=border, caption=embs, z=np.random.normal(size=(64, 100)))[0]

    for i, r in enumerate(result):
        save_image(r, emb=embs[i], folder="samples/random")
    # save_batch(result[0], 'samples', img_tag='random')


def noise_samples(dcgan, exemple_it):
    k = next(exemple_it)
    border = to_float(k[0])
    caption_emb = get_embedding(k[1])

    z0 = np.random.normal(size=(100,))
    z1 = np.random.normal(size=(100,))

    z = interpolate_noise(z0, z1)

    # make the batch
    batch_border = _make_batch(border)
    batch_captions = _make_batch(caption_emb)

    result = dcgan.eval(image=batch_border, caption=batch_captions, z=z)[0]
    for i, r in enumerate(result):
        save_image(r, emb=caption_emb, folder="samples/noise_interpolate")


def wrong_captions_samples(dcgan, exemple_it):
    k0 = _get_batch(exemple_it)
    k1 = _get_batch(exemple_it)
    batch_border = to_float(k0[0])
    batch_captions = get_embedding(k1[1])
    batch_z = _make_batch(np.random.normal(size=(100,)))

    result = dcgan.eval(image=batch_border, caption=batch_captions, z=batch_z)
    save_batch(result[0], 'samples', img_tag='wrong_captions')


def caption_variation(dcgan, exemple_it):
    k0 = _get_batch(exemple_it)
    k1 = next(exemple_it)
    batch_border = _make_batch(to_float(k1[0]))
    batch_captions = get_embedding(k0[1])
    batch_z = _make_batch(np.random.normal(size=(100,)))

    result = dcgan.eval(image=batch_border, caption=batch_captions, z=batch_z)
    save_batch(result[0], 'samples', img_tag='caption_variation')


def caption_random_combo(dcgan, exemple_it):
    k1 = next(exemple_it)
    batch_border = _make_batch(to_float(k1[0]))
    batch_captions = np.array([get_embedding(k1[1], combo=True) for _ in range(64)])
    batch_z = _make_batch(np.random.normal(size=(100,)))

    result = dcgan.eval(image=batch_border, caption=batch_captions, z=batch_z)
    save_batch(result[0], 'samples', img_tag='caption_random_combo')



if __name__ == '__main__':
  tf.app.run()
