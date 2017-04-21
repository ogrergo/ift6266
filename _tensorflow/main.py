from test import FLAGS, Model
import tensorflow as tf
import pprint
import os

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
      dcgan = Model(sess)

      # tf.show_all_variables()
      if FLAGS.is_train:
          dcgan.train()
      else:
          if not dcgan.load(FLAGS.checkpoint_dir):
              raise Exception("[!] Train a model first, then run test mode")


    # visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
