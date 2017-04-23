import glob
import os

import tensorflow as tf
import numpy as np
import time

from dask.array.chunk import mean

from op import *

from utils import *
from utils_old import save_images

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_integer("train_size", np.inf, "The size of train images")
flags.DEFINE_integer("batch_size", 64, "The size of batch images")
flags.DEFINE_integer("output_size", 32, "The size of the output images to produce")
flags.DEFINE_integer("input_size", 64, "The size of the input images")

flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise")
flags.DEFINE_integer("border_dim", 100, "Dimension of the border condition noise")

flags.DEFINE_integer("nb_fc", 2048, "1st layer of fully connected in generator.")
flags.DEFINE_integer("nb_filters_g", 128, "Number of filter in the last layer of generator")
flags.DEFINE_integer("nb_filters_d", 128, "Number of filter in the 1st layer of discriminator")

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

        self.embeddings_real = tf.placeholder(tf.float32, [None, 1024], name="embeddings")
        self.embeddings_fake = tf.placeholder(tf.float32, [None, 1024], name="embeddings")

        mask = np.ones((FLAGS.batch_size, 64, 64, 3), dtype='float32')
        mask[:, 16:48, 16:48, :] = 0

        self.input_border = tf.multiply(self.input_true, tf.constant(mask, name="center_mask"))

        # input_sample = tf.placeholder(tf.float32,
        #     [None, FLAGS.input_size, FLAGS.input_size, FLAGS.c_dim],
        #     name="input_sample")

        self.G = self.generator(self.z, border=self.input_border, embeddings=self.embeddings_real)
        self.G_sampler = self.generator(self.z, reuse=True, border=self.input_border, embeddings=self.embeddings_real)

        self.D, self.D_logits = self.discriminator(self.input_true, reuse=False,
                                                   embeddings=self.embeddings_real)

        self.D_sample, self.D_logits_sample = self.discriminator(self.G, reuse=True,
                                                                 embeddings=self.embeddings_real)

        self.D_fake_captions, self.D_logits_fake_captions = self.discriminator(self.input_true,
                                                                               reuse=True,
                                                                               embeddings=self.embeddings_fake)


        # random label mean
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.random_normal(shape=tf.shape(self.D), mean=.9, stddev=.1)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_sample, labels=tf.zeros_like(self.D_sample)))

        self.d_loss_real_wrong_caption = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake_captions,
                                                    labels=tf.zeros_like(self.D_fake_captions)))


        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_sample, labels=tf.ones_like(self.D_sample)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss_fake_captions_sum = scalar_summary("d_loss_fake_captions", self.d_loss_real_wrong_caption)
            

        self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_real_wrong_caption

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_sample)

        self.G_sum = image_summary("G", self.G, max_outputs=10)

        self.saver = tf.train.Saver()

    def discriminator(self, x, reuse=False, embeddings=None):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()


            h0 = lrelu(conv2d(x, FLAGS.nb_filters_d, name='d_h0_conv'))

            h1 = lrelu(self.d_bn1(conv2d(h0, FLAGS.nb_filters_d*2, name='d_h1_conv')))

            h2 = lrelu(self.d_bn2(conv2d(h1, FLAGS.nb_filters_d*4, name='d_h2_conv')))

            y_emb_conv = tf.reshape(embeddings, [FLAGS.batch_size, 8, 8, 16])
            h2 = tf.concat([h2, y_emb_conv], 3)

            h3 = lrelu(self.d_bn3(conv2d(h2, FLAGS.nb_filters_d*8, name='d_h3_conv')))

            shape = h3.get_shape()
            r = tf.reshape(h3, [FLAGS.batch_size, int(shape[1] * shape[2] * shape[3])])

            r = tf.concat([r, embeddings], 1)

            h4 = tf.reshape(linear(r, 1, 'd_h3_lin'), [-1])
            tf.assert_equal(tf.shape(h4), (FLAGS.batch_size,))

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, reuse=False, border=None, embeddings=None):
        ## dim embeddings (None, 1024)

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
                    conv2d(border, FLAGS.nb_filters_g//2, name="g_b0_conv"), train=train)), s_h2)

                b1 = avg_pooling(tf.nn.relu(self.g_bn4(
                    conv2d(b0, FLAGS.nb_filters_g, name="g_b1_conv"), train=train)), s_h4)

                y = linear(tf.reshape(b1, [FLAGS.batch_size, s_h4 * s_w4 * FLAGS.nb_filters_g]), 100, "g_border_lin")

                y_emb = linear(embeddings, 300, "g_project_emb")

                z = concat([z, y, y_emb], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, FLAGS.nb_fc, 'g_h0_lin'), train=train))
                h0 = concat([h0, tf.nn.relu(y), y_emb], 1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0, FLAGS.nb_filters_g * 2 * s_h4 * s_w4, 'g_h1_lin'), train=train))
                
                s = int(h1.get_shape()[1]) // (s_h4**2)
                h1 = tf.reshape(h1, [FLAGS.batch_size, s_h4, s_w4, s])

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

        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.d_loss_fake_captions_sum])

        self.writer = SummaryWriter("./logs", self.session.graph)

        sample_z = np.random.uniform(-1, 1, size=(FLAGS.batch_size, FLAGS.z_dim))

        sample_inputs = load_images(data[0:FLAGS.batch_size], middle=False)
        sample_emb = get_embeddings(data[0:FLAGS.batch_size])

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

                batch_emb = get_embeddings(filelist=batch_files)
                batch_emb_fake = get_embeddings(batch_size=FLAGS.batch_size)

                # Update D network
                _, summary_str = self.session.run([d_optim, self.d_sum],
                                               feed_dict={self.input_true: batch,
                                                          self.z: batch_z,
                                                          self.embeddings_real: batch_emb,
                                                          self.embeddings_fake: batch_emb_fake})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.session.run([g_optim, self.g_sum],
                                               feed_dict={self.input_true: batch,
                                                          self.z: batch_z,
                                                          self.embeddings_real:batch_emb})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.session.run([g_optim, self.g_sum],
                                               feed_dict={self.input_true: batch,
                                                          self.z: batch_z,
                                                          self.embeddings_real:batch_emb})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.input_true: batch,
                                                   self.z: batch_z,
                                                   self.embeddings_real:batch_emb})

                errD_real = self.d_loss_real.eval({self.input_true: batch,
                                                   self.embeddings_real:batch_emb})

                errD_real_fake_caption = self.d_loss_real_wrong_caption.eval({
                    self.input_true: batch,
                    self.embeddings_fake: batch_emb_fake
                })

                errG = self.g_loss.eval({self.input_true: batch,
                                         self.z: batch_z,
                                         self.embeddings_real:batch_emb})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real + errD_real_fake_caption, errG))

                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.session.run(
                            [self.G_sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.input_true: sample_inputs,
                                self.embeddings_real: sample_emb
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
