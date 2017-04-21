import glob
import os

import tensorflow as tf
import numpy as np
import time

from dask.array.chunk import mean

from _tensorflow.op import *

from _tensorflow.utils import load_images, get_dataset_files
from _tensorflow.utils_old import save_images
from ops import conv_cond_concat

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_integer("train_size", np.inf, "The size of train images")
flags.DEFINE_integer("batch_size", 64, "The size of batch images")
flags.DEFINE_integer("output_size", 32, "The size of the output images to produce")
flags.DEFINE_integer("input_size", 64, "The size of the input images")

flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise")
flags.DEFINE_integer("border_dim", 100, "Dimension of the border condition noise")

flags.DEFINE_integer("nb_fc", 1024, "")
flags.DEFINE_integer("nb_filters_g", 64, "Number of filter in the last layer of generator")
flags.DEFINE_integer("nb_filters_d", 64, "Number of filter in the 1st layer of discriminator")

flags.DEFINE_string("dataset", "ift_image_only", "The name of dataset")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples")

flags.DEFINE_string("model_type", "dcgan", "model type")


flags.DEFINE_boolean("is_train", True, "True for training, False for testing")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing")
FLAGS = flags.FLAGS

class Model():
    def __init__(self, session):
        self.session = session

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # if not self.y_dim:
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        # if not self.y_dim:
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.build_model()

    def build_model(self):
        ## build cGAN model
        # self.y = tf.placeholder(tf.float32, [None, FLAGS.y_dim], name='y')
        self.z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.input_true = tf.placeholder(tf.float32,
            [None, FLAGS.input_size, FLAGS.input_size, FLAGS.c_dim],
            name="input_true")

        mask = np.ones((FLAGS.batch_size, 64, 64, 3), dtype='float32')
        mask[:, 16:48, 16:48, :] = 0

        self.input_border = tf.multiply(self.input_true, tf.constant(mask, name="center_mask"))

        # input_sample = tf.placeholder(tf.float32,
        #     [None, FLAGS.input_size, FLAGS.input_size, FLAGS.c_dim],
        #     name="input_sample")

        self.G = self.generator(self.z, border=self.input_border)
        self.G_sampler = self.generator(self.z, reuse=True, border=self.input_border)

        self.D, self.D_logits = self.discriminator(self.input_true, reuse=False)
        self.D_sample, self.D_logits_sample = self.discriminator(self.G, reuse=True)

        # random label mean
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.random_normal(shape=tf.shape(self.D), mean=.9, stddev=.1)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_sample, labels=tf.zeros_like(self.D_sample)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_sample, labels=tf.ones_like(self.D_sample)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_sample)
        self.d_logits_sum = histogram_summary("d_logits", self.D_logits)
        self.d_logits_sampler_sum = histogram_summary("d_logits_sampler", self.D_logits_sample)

        self.G_sum = image_summary("G", self.G, max_outputs=10)

        self.saver = tf.train.Saver()

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(x, FLAGS.nb_filters_d, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, FLAGS.nb_filters_d*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, FLAGS.nb_filters_d*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, FLAGS.nb_filters_d*8, name='d_h3_conv')))

            shape = h3.get_shape()
            r = tf.reshape(h3, [FLAGS.batch_size, int(shape[1] * shape[2] * shape[3])])


            h4 = tf.reshape(linear(r, 1, 'd_h3_lin'), [-1])
            tf.assert_equal(tf.shape(h4), (FLAGS.batch_size,))

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, reuse=False, border=None):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            train = not reuse

            if border is not None:
                # border dim is (64, 64, 64, 3)
                s_h, s_w = FLAGS.output_size, FLAGS.output_size
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                ## build conv network

                b0 = avg_pooling(tf.nn.relu(self.g_bn3(
                    conv2d(border, FLAGS.nb_filters_g//4, name="g_b0_conv"), train=train)), s_h2)


                b1 = avg_pooling(tf.nn.relu(self.g_bn4(
                    conv2d(border, FLAGS.nb_filters_g//2, name="g_b1_conv"), train=train)), s_h4)

                y = linear(tf.reshape(b1, [FLAGS.batch_size, s_h4 * s_w4 * FLAGS.nb_filters_g//2]), 100, "g_border_lin")
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, FLAGS.nb_fc, 'g_h0_lin'), train=train))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0, FLAGS.nb_filters_g * 2 * s_h4 * s_w4, 'g_h1_lin'), train=train))
                h1 = tf.reshape(h1, [FLAGS.batch_size, s_h4, s_w4, FLAGS.nb_filters_g * 2])

                h1 = concat([h1, b1], 3)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [FLAGS.batch_size, s_h2, s_w2, FLAGS.nb_filters_g * 2], name='g_h2'), train=train))

                h2 = concat([h2, b0], 3)

                h3 = tf.nn.tanh(deconv2d(h2, [FLAGS.batch_size, s_h, s_w, FLAGS.c_dim], name='g_h3'))
                center = tf.pad(h3, [[0, 0],[16, 16], [16, 16], [0, 0]], "CONSTANT")

                out = border + center
                return out

            else:
                s_h, s_w = FLAGS.output_size, FLAGS.output_size
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, FLAGS.nb_filters_g*8*s_h16*s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, FLAGS.nb_filters_g * 8])

                h0 = tf.nn.relu(self.g_bn0(h0, train=train))

                h1 = deconv2d(h0, [FLAGS.batch_size, s_h8, s_w8, FLAGS.nb_filters_g*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=train))

                h2 = deconv2d(h1, [FLAGS.batch_size, s_h4, s_w4, FLAGS.nb_filters_g*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=train))

                h3 = deconv2d(h2, [FLAGS.batch_size, s_h2, s_w2, FLAGS.nb_filters_g*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=train))

                h4 = deconv2d(h3, [FLAGS.batch_size, s_h, s_w, FLAGS.c_dim], name='g_h4')

                return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        return "{}_{}_{}".format("IFT6266", FLAGS.batch_size, FLAGS.model_type)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train(self):
        """Train DCGAN"""

        print(" [*] Loading dataset")
        data = get_dataset_files()
        # np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)


        tf.global_variables_initializer().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])

        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum,
                                    self.d_logits_sum, self.d_logits_sampler_sum])
        self.writer = SummaryWriter("./logs", self.session.graph)

        sample_z = np.random.uniform(-1, 1, size=(FLAGS.batch_size, FLAGS.z_dim))

        sample_inputs = load_images(data[0:FLAGS.batch_size], middle=False)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(FLAGS.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(FLAGS.epoch):
            data = get_dataset_files()

            batch_idxs = min(len(data), FLAGS.train_size) // FLAGS.batch_size

            for idx in range(0, batch_idxs):
                batch_files = data[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
                batch = load_images(batch_files, middle=False)

                batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.session.run([d_optim, self.d_sum],
                                               feed_dict={self.input_true: batch, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.session.run([g_optim, self.g_sum],
                                               feed_dict={self.input_true: batch, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.session.run([g_optim, self.g_sum],
                                               feed_dict={self.input_true: batch, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.input_true: batch, self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.input_true: batch})
                errG = self.g_loss.eval({self.input_true: batch, self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.session.run(
                            [self.G_sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.input_true: sample_inputs,
                            },
                        )
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                                    './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(FLAGS.checkpoint_dir, counter)
