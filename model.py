import tensorflow as tf
import numpy as np
import rnn_helper as rnn
import config
import func
import utils
import os

class Model(object):
    def __init__(self, vocab_size, ckpt_folder=None, name='model'):
        self.name = name
        self.ckpt_folder = ckpt_folder
        self.vocab_size = vocab_size
        if self.ckpt_folder is not None:
            utils.mkdir(self.ckpt_folder)
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        with tf.variable_scope(self.name, initializer=initializer):
            self.initialize()


    def initialize(self):
        self.create_input()
        self.create_embedding()
        self.create_passage_encoder()
        self.create_question_encoder()
        self.create_similarity()
        self.create_loss()
        self.create_optimizer()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        total, vc = self.trainable_parameters()
        print('trainable parameters: {}'.format(total))
        for name, count in vc.items():
            print('{}: {}'.format(name, count))


    def feed(self, passage, question, label, keep_prob):
        feed_dict = {
            self.input_passage: passage,
            self.input_question: question,
            self.input_question_label: label,
            self.input_keep_prob: keep_prob
        }
        return feed_dict


    def create_input(self):
        with tf.variable_scope('input'):
            self.input_passage = tf.placeholder(tf.int32, shape=[None, None], name='passage')
            self.passage_mask, self.passage_length = func.tensor_to_mask(self.input_passage)

            self.input_question = tf.placeholder(tf.int32, shape=[None, None, None], name='question')
            self.input_question_label = tf.placeholder(tf.float32, shape=[None, None], name='question_label')#0 or 1
            self.batch_size = tf.shape(self.input_question)[0]
            self.max_num_question = tf.shape(self.input_question)[1]
            self.max_question_len = tf.shape(self.input_question)[2]
            self.target_question = tf.reshape(self.input_question, shape=[self.batch_size*self.max_num_question, self.max_question_len])
            self.target_question_label = tf.reshape(self.input_question_label, shape=[-1])
            self.question_mask, self.question_length = func.tensor_to_mask(self.target_question)
            #self.num_question_mask, self.num_question = func.tensor_to_mask(self.question_length)
            self.input_keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    def create_embedding(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, config.embedding_dim])
            self.passage_embedding = tf.nn.embedding_lookup(self.embedding, self.input_passage, name='passage_embedding')
            self.question_embedding = tf.nn.embedding_lookup(self.embedding, self.target_question, name='question_embedding')
            tf.summary.histogram('embedding/embedding', self.embedding)


    def create_passage_encoder(self):
        with tf.variable_scope('passage_encoder'):
            bi_output, bi_state = self.multi_bilstm(
                self.passage_embedding,
                self.passage_length,
                config.encoder_hidden_dim,
                config.num_passage_encoder_layers,
                config.num_passage_residual_layers)
            self.passage_encoder_output = tf.concat(bi_output, -1, name='output')
        #    self.passage_encoder_state = self.dense_output(bi_output, self.passage_mask)
            self.passage_encoder_state = self.dense_state(bi_state)
            tf.summary.histogram('passage_encoder/output', self.passage_encoder_output)
            tf.summary.histogram('passage_encoder/state', self.passage_encoder_state)


    def create_question_encoder(self):
        with tf.variable_scope('question_encoder'):
            bi_output, bi_state = self.multi_bilstm(
                self.question_embedding,
                self.question_length,
                config.encoder_hidden_dim,
                config.num_question_encoder_layers,
                config.num_question_residual_layers)
            self.question_encoder_output = tf.concat(bi_output, -1, name='output')
            #self.question_encoder_state = self.dense_output(bi_output, self.question_mask)
            self.question_encoder_state = self.dense_state(bi_state)
            tf.summary.histogram('question_encoder/output', self.question_encoder_output)
            tf.summary.histogram('question_encoder/state', self.question_encoder_state)


    def create_similarity(self):
        with tf.variable_scope('similarity'):
            tiled_passage = tf.tile(self.passage_encoder_state, [self.max_num_question, 1])
            self.similarity = tf.reduce_sum(self.question_encoder_state * tiled_passage, -1, name='similarity')
            self.norm_similarity = tf.sigmoid(self.similarity)
            #self.similarity = tf.clip_by_value(self.similarity, -20, 20, name='similarity')


    def create_loss(self):
        with tf.variable_scope('loss'):
            target_mask = tf.to_float(self.question_length > 0)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.similarity, labels=self.target_question_label) * target_mask
            self.loss = tf.reduce_sum(loss) / tf.reduce_sum(target_mask)
            tf.summary.scalar('loss/loss', self.loss)
            

    def create_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate=1E-3)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)


    def trainable_parameters(self):
        total_parameters = 0
        vc = {}
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            vc[variable.name] = variable_parameters
        return total_parameters, vc


    def multi_bilstm(self, input, length, hidden_dim, num_layers, num_residual_layers):
        fw_cell = rnn.create_rnn_cell('lstm', hidden_dim, num_layers, num_residual_layers, self.input_keep_prob)
        bw_cell = rnn.create_rnn_cell('lstm', hidden_dim, num_layers, num_residual_layers, self.input_keep_prob)
        bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, dtype=tf.float32, sequence_length=length)
        return bi_output, bi_state


    def dense_state(self, bi_state, name='dense_state'):
        state_list = []
        for layer_id in range(len(bi_state[0])):
            for direction in range(2):
                for ch in range(2):
                    state_list.append(bi_state[direction][layer_id][ch])
        return tf.layers.dense(tf.concat(state_list, -1), config.dense_vector_dim, name=name)


    def dense_output(self, bi_output, mask, name='dense_output'):
        forward = bi_output[0]
        backward = bi_output[1]
        output = tf.concat([forward, backward], -1) * tf.expand_dims(mask, -1)
        state = tf.reduce_sum(output, 1)
        return tf.layers.dense(state, config.dense_vector_dim, name=name)
        

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_folder)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('MODEL LOADED.')
        else:
            sess.run(tf.global_variables_initializer())


    def save(self, sess):
        self.saver.save(sess, os.path.join(self.ckpt_folder, 'model.ckpt'))


    def summarize(self, writer):
        self.summary = tf.summary.merge_all()


if __name__ == '__main__':
    model = Model(config.char_vocab_size)