import numpy as np
import preprocess
import utils
import config
from data import TrainFeeder, align2d, Dataset


def run_epoch(sess, model, feeder, writer):
    feeder.prepare('dev')
    while not feeder.eof():
        aids, qv, av, kb = feeder.next(32)
        feed = model.feed(aids, qv, av, kb)
        answer_logit, question_logit = sess.run([model.answer_logit, model.question_logit], feed_dict=feed)
        question = [id for id, v in enumerate(question_logit) if v >= 0]
        answer = [id for id, v in enumerate(answer_logit) if v >= 0]
        return question, answer
        

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class Evaluator(TrainFeeder):
    def __init__(self, dataset=None):
        super(Evaluator, self).__init__(Dataset() if dataset is None else dataset)


    def create_feed(self, answer, question):
        #question = question.split(' ')
        #answer = answer.split(' ')
        aids = self.sent_to_id(answer)
        qids = self.sent_to_id(question)
        qv, _ = self.label_qa(question)
        st = self.seq_tag(question, answer)
        return aids, qids, qv, st, 1.0


    def predict(self, sess, model, answer, question):
        aids, qids, qv, av, kb = self.create_feed(answer, question)
        feed = model.feed([aids], [qids], [qv], [av], kb)
        question_logit = sess.run(model.question_logit, feed_dict=feed)
        predict_question = self.decode_logit(question_logit[0])
        print('==================================================')
        print('answer', ' '.join(answer))
        print('---------------------------------------------------')
        print('question', ' '.join(question))
        print('predict question', predict_question)
        #print('question score', [v for _,v in qids])
        #print('answer score', ['{}:{:>.4f}'.format(w,x) for w,x in zip(answer, answer_logit[0])])


    def evaluate(self, sess, model):
        self.prepare('dev')
        pids, qids, labels, kb = self.next()
        feed = model.feed(pids, qids, labels, kb)
        loss = sess.run(model.loss, feed_dict=feed)
        return loss


if __name__ == '__main__':
    from model import Model
    import tensorflow as tf
    evaluator = Evaluator()
    model = Model(evaluator.dataset.qi2c, config.checkpoint_folder, False)
    with tf.Session() as sess:
        model.restore(sess)
        #evaluator.evaluate(sess, model, 'The cat sat on the mat', 'what is on the mat')
        evaluator.prepare('dev')
        for question, answer in evaluator.data[:10]:
            evaluator.predict(sess, model, answer, question)