import tensorflow as tf
import numpy as np
import rnn_helper as rnn
import cnn_helper as cnn
import config
import func
import utils
import os

class Model(object):
    def __init__(self, vocab_size, ckpt_folder=None, name='discriminator'):
        self.name = name
        self.ckpt_folder = ckpt_folder
        self.vocab_size = vocab_size
        self.initializer = tf.random_uniform_initializer(-0.05, 0.05)
        with tf.variable_scope(self.name, initializer=self.initializer):
            self.initialize()


    def initialize(self):
        self.create_input()
        self.create_embedding()
        self.create_passage_encoder()
        self.create_question_encoder()
        self.create_similarity()
        self.create_loss()
        total, vc = self.trainable_parameters()
        print('trainable parameters: {}'.format(total))
        for name, count in vc.items():
            print('{}: {}'.format(name, count))


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
        if config.using_cnn:
            self.passage_encoder_state = self.conv_encoder(
                self.passage_embedding,
                self.passage_length,                
                config.num_passage_encoder_layers,
                'passage_encoder')
        else:
            self.passage_encoder_state = self.nlstm_encoder(
                self.passage_embedding,
                self.passage_length,
                config.num_passage_encoder_layers,
                config.num_passage_residual_layers,
                'passage_encoder')


    def create_question_encoder(self):
        if config.using_cnn:
            self.question_encoder_state = self.create_question_encoding(self.question_embedding, self.question_length)


    def create_question_encoding(self, input, length):
        if config.using_cnn:
            return self.conv_encoder(
                input, length,
                config.num_question_encoder_layers,
                'question_encoder')
        else:
            return self.nlstm_encoder(
                input, length,
                config.num_question_encoder_layers,
                config.num_question_residual_layers,
                'question_encoder')


    def conv_encoder(self, inputs, length, num_layers, scope):
        mask = tf.expand_dims(tf.sequence_mask(length, dtype=tf.float32), -1)
        outputs0 = cnn.conv(inputs*mask, num_layers, config.encoder_kernel_size0, config.dense_vector_dim//2, self.input_keep_prob, scope+'.output0')
        outputs1 = cnn.conv(inputs*mask, num_layers, config.encoder_kernel_size1, config.dense_vector_dim//2, self.input_keep_prob, scope+'.output1')
        outputs = tf.concat([outputs0, outputs1], -1)
        gate = tf.sigmoid(cnn.conv(inputs*mask, num_layers, config.encoder_gate_size, config.dense_vector_dim, self.input_keep_prob, scope+'.gate'))
        #return self.self_attention(outputs * gate, length, scope)
        return tf.reduce_sum(outputs*gate, 1)


    def nlstm_encoder(self, inputs, length, num_encoder_layers, num_residual_layers, scope):
        with tf.variable_scope(scope):
            _, bi_state = self.multi_bilstm(
                inputs, length, config.encoder_hidden_dim,
                num_encoder_layers, num_residual_layers)
            #encoder_output = tf.concat(bi_output, -1, name='output')
            #encoder_state = self.dense_output(bi_output, self.question_mask)
            encoder_state = self.dense_state(bi_state)
            tf.summary.histogram('{}/state'.format(scope), encoder_state)
            return encoder_state


    def self_attention(self, inputs, length, scope):
        with tf.variable_scope(scope):
            mask = tf.sequence_mask(length, dtype=tf.float32)
            self_att, _ = func.dot_attention(inputs, inputs, mask, config.encoder_hidden_dim, self.input_keep_prob)
            self_match, _ = func.rnn('gru', self_att, length, config.encoder_hidden_dim, 1, self.input_keep_prob)
            ws_answer = tf.get_variable(name='ws_answer', shape=[config.encoder_hidden_dim, 1])
            logit = tf.einsum('aij,jk->aik', self_match, ws_answer)
            alpha = tf.sigmoid(logit, name='alpha')
            return tf.reduce_sum(alpha*inputs, 1, name='dense_vector')


    def create_similarity(self):
        with tf.variable_scope('similarity'):
            tiled_passage = tf.tile(self.passage_encoder_state, [self.max_num_question, 1])
            self.similarity = tf.reduce_sum(self.question_encoder_state * tiled_passage, -1, name='similarity')
            self.norm_similarity = tf.sigmoid(self.similarity)


    def create_loss(self):
        with tf.variable_scope('loss'):
            target_mask = tf.to_float(self.question_length > 0)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.similarity, labels=self.target_question_label) * target_mask
            self.loss = tf.reduce_sum(loss) / tf.reduce_sum(target_mask)
            tf.summary.scalar('loss/loss', self.loss)


    def trainable_parameters(self):
        total_parameters = 0
        vc = {}
        for variable in tf.trainable_variables(self.name):
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


if __name__ == '__main__':
    model = Model(13000)