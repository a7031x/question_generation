import tensorflow as tf
import numpy as np
import config
import utils
import os
from model_generator import Model as Generator
from model_discriminator import Model as Discriminator

class Model(object):
    def __init__(self, ci2n, ckpt_folder):
        self.discriminator = Discriminator(len(ci2n))
        self.generator = Generator(ci2n, self.discriminator.embedding, self.discriminator.initializer)
        self.global_step = tf.Variable(0, trainable=False)
        self.ckpt_folder = ckpt_folder
        if self.ckpt_folder is not None:
            utils.mkdir(self.ckpt_folder)
        self.discriminator_optimizer, self.discriminator_loss = self.create_discriminator_optimizer(True)
        self.discriminator_optimizer0, self.discriminator_loss0 = self.create_discriminator_optimizer(False)
        self.generator_optimizer, self.generator_loss = self.create_generator_optimizer()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)


    def feed_discriminator(self, passage, question, label, keep_prob):
        feed_dict = {
            self.discriminator.input_passage: passage,
            self.discriminator.input_question: question,
            self.discriminator.input_question_label: label,
            self.discriminator.input_keep_prob: keep_prob,
            self.generator.input_word: passage,
            self.generator.input_keep_prob: 1
        }
        return feed_dict


    def feed_generator(self, passage, keep_prob):
        feed_dict = {
            self.discriminator.input_passage: passage,
            self.discriminator.input_keep_prob: 1,
            self.generator.input_word: passage,
            #self.generator.input_label_question_vector: question_vector,
            #self.generator.input_label_answer: answer_tag,
            self.discriminator.input_question: [[[0]]],
            self.discriminator.input_question_label: [[1]],
            self.generator.input_keep_prob: keep_prob
        }
        return feed_dict


    def create_generator_loss(self, target):
        generator_logit = tf.nn.softmax(self.generator.question_logit)
        generator_ids = tf.argmax(generator_logit, -1)
        generator_mask = tf.to_float(tf.not_equal(generator_ids, config.EOS_ID))
        generator_embed = tf.einsum('bij,jk->bik', generator_logit, self.discriminator.embedding) * tf.expand_dims(generator_mask, -1)
        generator_length = tf.reduce_sum(generator_mask, -1)
        generator_state = self.discriminator.create_question_encoding(generator_embed, generator_length)
        similarity = tf.reduce_sum(generator_state*self.discriminator.passage_encoder_state, -1)
        self.norm_similarity = tf.sigmoid(similarity)
        label = tf.fill(tf.shape(similarity), np.float(target))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=similarity, labels=label)
        return tf.reduce_sum(loss)


    def create_discriminator_optimizer(self, with_generator_loss):
        discriminator_loss = self.discriminator.loss
        loss = discriminator_loss
        if with_generator_loss:
            generator_loss = self.create_generator_loss(0)
            loss += generator_loss
            tf.summary.scalar('discriminator/generator/loss', generator_loss)
        tf.summary.scalar('discriminator/discriminator/loss', discriminator_loss)
        tf.summary.scalar('discriminator/loss', loss)
        return self.create_optimizer(loss, self.discriminator.name), loss


    def create_generator_optimizer(self):
        loss = self.create_generator_loss(1)
        tf.summary.scalar('generator/loss', loss)
        return self.create_optimizer(loss, self.generator.name), loss


    def create_optimizer(self, loss, scope):
        opt = tf.train.AdamOptimizer(learning_rate=1E-4)
        grads = opt.compute_gradients(loss, var_list=tf.trainable_variables(scope))
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, 5.0)
        return opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)


    def summarize(self):
        self.summary = tf.summary.merge_all()

        
    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_folder)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print('MODEL LOADED.')
        else:
            sess.run(tf.global_variables_initializer())


    def save(self, sess):
        self.saver.save(sess, os.path.join(self.ckpt_folder, 'model.ckpt'))