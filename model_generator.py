import tensorflow as tf
import tensorflow.contrib.seq2seq as ts
from tensorflow.python.layers import core as layers_core
import rnn_helper as rnn
import numpy as np
import utils
import func
import config
import os

class Model(object):
    def __init__(self, ci2n, embedding, initializer, name='generator'):
        self.name = name
        self.vocab_size = len(ci2n)
        qww = [0] * self.vocab_size
        for i,c in ci2n.items():
            qww[i] = c
        qwm = qww[80]
        qww = [c if c < qwm else qwm for c in qww]
        qww = np.array(qww) ** 0.5
        self.word_weight = 5 - 4.7 * qww / np.max(qww)
        self.embedding = embedding
        print(self.word_weight[78:100])
        with tf.variable_scope(self.name, initializer=initializer):
            self.initialize()


    def initialize(self):
        self.create_inputs()
        self.create_embeddings()
        self.create_encoder()
        self.create_selfmatch()
        self.create_decoder()
        total, vc = self.trainable_parameters()
        print('trainable parameters: {}'.format(total))
        for name, count in vc.items():
            print('{}: {}'.format(name, count))


    def create_inputs(self):
        with tf.name_scope('input'):
            self.input_word = tf.placeholder(tf.int32, shape=[None, None], name='word')
            self.input_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.batch_size = tf.shape(self.input_word)[0]
            self.mask, self.length = func.tensor_to_mask(self.input_word)
            self.input_label_answer = tf.placeholder(tf.float32, shape=[None, None], name='label_answer')
            self.input_label_question_vector = tf.placeholder(tf.float32, shape=[None, self.vocab_size], name='label_question_vector')


    def feed(self, aids, qids=None, qv=None, st=None, keep_prob=1.0):
        feed_dict = {
            self.input_word: aids,
            self.input_keep_prob: keep_prob
        }
        if qv is not None:
            for v in qv:
                v[config.EOS_ID] = 1
            feed_dict[self.input_label_question_vector] = qv
        if st is not None:
            feed_dict[self.input_label_answer] = st
        return feed_dict


    def create_embeddings(self):
        with tf.name_scope('embedding'):
            self.emb = tf.nn.embedding_lookup(self.embedding, self.input_word, name='emb')
            tf.summary.histogram('embedding/emb', self.emb)


    def create_encoder(self):
        with tf.name_scope('encoder'):
            fw_cell = rnn.create_rnn_cell('lstm', config.encoder_hidden_dim, config.num_passage_encoder_layers, config.num_passage_residual_layers, self.input_keep_prob)
            bw_cell = rnn.create_rnn_cell('lstm', config.encoder_hidden_dim, config.num_passage_encoder_layers, config.num_passage_residual_layers, self.input_keep_prob)
            bi_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.emb, dtype=tf.float32, sequence_length=self.length, swap_memory=True)
            self.encoder_output = tf.concat(bi_output, -1)
            encoder_state = []
            for layer_id in range(config.num_passage_encoder_layers):
                encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                encoder_state.append(bi_encoder_state[1][layer_id])  # backward
            self.encoder_state = tuple(encoder_state)
            tf.summary.histogram('encoder/encoder_output', self.encoder_output)
            tf.summary.histogram('encoder/encoder_state', self.encoder_state)


    def create_selfmatch(self):
        with tf.name_scope('selfmatch'), tf.variable_scope('selfmatch'):
            self_att, _ = func.dot_attention(self.encoder_output, self.encoder_output, self.mask, config.encoder_hidden_dim, self.input_keep_prob)
            selfmatch, _ = func.rnn('gru', self_att, self.length, config.encoder_hidden_dim, 1, self.input_keep_prob)
            tf.summary.histogram('self_attention/self_match', selfmatch)
            self.ws_answer = tf.get_variable(name='ws_answer', shape=[config.encoder_hidden_dim, 1])
            self.answer_logit = tf.einsum('aij,jk->aik', selfmatch, self.ws_answer)
            self.answer_alpha = tf.sigmoid(self.answer_logit, name='answer_alpha')
            self.answer_logit = tf.squeeze(self.answer_logit, [-1])
            #self.answer_logit = tf.squeeze(self.answer_logit, [-1])
            self.answer_vector = tf.reduce_sum(self.answer_alpha*self.encoder_output, 1)
            tf.summary.histogram('self_attention/answer_logit', self.answer_logit)
            tf.summary.histogram('self_attention/answer_alpha', self.answer_alpha)
            tf.summary.histogram('self_attention/answer_vector', self.answer_vector)


    def create_decoder(self):
        with tf.variable_scope('decoder/output_projection'):
            output_layer = layers_core.Dense(self.vocab_size, use_bias=False, name='output_projection')
        with tf.name_scope('decoder'), tf.variable_scope('decoder') as decoder_scope:
            memory = self.encoder_output
            source_sequence_length = self.length
            encoder_state = self.encoder_state
            batch_size = self.batch_size

            attention_mechanism = ts.LuongAttention(config.decoder_hidden_dim, memory, source_sequence_length, scale=True)
            cell = rnn.create_rnn_cell('lstm', config.decoder_hidden_dim, config.num_decoder_rnn_layers, config.num_decoder_residual_layers, self.input_keep_prob)
            cell = ts.AttentionWrapper(cell, attention_mechanism,
                attention_layer_size=config.decoder_hidden_dim,
                alignment_history=False,
                output_attention=True,
                name='attention')

            decoder_initial_state = cell.zero_state(batch_size, tf.float32)
            #start_tokens = tf.fill([self.batch_size], config.SOS_ID)
            #target_input = tf.concat([tf.expand_dims(start_tokens, -1), self.input_label_question], 1)
            vector_input = tf.tile(tf.expand_dims(self.answer_vector, 1), [1, config.max_question_len, 1])
            #decoder_emb = tf.nn.embedding_lookup(self.question_embedding, vector_inp)
            helper = ts.TrainingHelper(vector_input, tf.fill([self.batch_size], config.max_question_len))
            decoder = ts.BasicDecoder(cell, helper, decoder_initial_state)
            output, self.final_context_state, _ = ts.dynamic_decode(decoder, swap_memory=True, scope=decoder_scope)
            self.question_logit = output_layer(output.rnn_output)
            tf.summary.histogram('decoder/question_logit', self.question_logit)

    
    def create_answer_tag_loss(self):
        with tf.name_scope('answer_tag_loss'):
            answer_loss = tf.nn.weighted_cross_entropy_with_logits(logits=self.answer_logit, targets=self.input_label_answer, pos_weight=3.0) * self.mask
            self.answer_tag_loss = tf.reduce_mean(tf.reduce_sum(answer_loss, -1))
            tf.summary.scalar('answer_tag_loss', self.answer_tag_loss)
            return self.answer_tag_loss


    def create_vector_loss(self):
        with tf.name_scope('vector_loss'):
            hardmax = ts.hardmax(self.question_logit)
            self.max_logit = hardmax * self.question_logit
            self.squeezed_logit = tf.reduce_sum(self.max_logit, 1)
            vector_weight = tf.cast(tf.reduce_sum(hardmax, 1) + self.input_label_question_vector > 0.0, tf.float32) * self.word_weight
            vector_loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_label_question_vector, logits=self.squeezed_logit, pos_weight=3.0) * vector_weight
            self.vector_loss = tf.reduce_mean(tf.reduce_sum(vector_loss, -1))
            tf.summary.scalar('vector_loss', self.vector_loss)
            return self.vector_loss


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


if __name__ == '__main__':
    from data import Dataset
    data = Dataset()
    embedding = tf.get_variable(name='embedding', shape=[len(data.ci2n), config.embedding_dim])
    model = Model(data.ci2n, embedding, tf.random_normal_initializer())
