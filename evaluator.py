import tensorflow as tf
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


def evaluate_discriminator(sess, model, feeder, writer, training, feed=None):
    pids, qids, labels, kb = feeder.next()
    if feed is None:
        feed = model.feed_discriminator(pids, qids, labels, kb)
    summary, global_step, _, loss, similarity, question_logit = sess.run(
        [
            model.summary, model.global_step, model.discriminator_optimizer if training else tf.no_op(), model.discriminator_loss,
            model.discriminator.norm_similarity, model.generator.question_logit
        ], feed_dict=feed)
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    pid, qid, sim, lab = pids[0], qids[0], similarity[:len(qids[0])], labels[0]
    passage = feeder.ids_to_sent(pid)
    questions = [feeder.ids_to_sent(q) for q in qid]
    print(passage)
    for q,s,l in zip(questions, sim, lab):
        if q:
            print(' {} {:>.4F}: {}'.format(l, s, q))
    generated_question = feeder.decode_logit(question_logit[0])
    print('generate: {}'.format(generated_question))
    return loss


def evaluate_generator(sess, model, feeder, writer, training, feed=None):
    pids, qids, _, kb = feeder.next()
    if feed is None:
        feed = model.feed_generator(pids, kb)
    summary, global_step, _, loss, similarity, question_logit = sess.run(
        [
            model.summary, model.global_step, model.generator_optimizer if training else tf.no_op(), model.generator_loss,
            model.norm_similarity, model.generator.question_logit
        ], feed_dict=feed)
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    for pid,qid,logit,sim in zip(pids, qids[:][0], question_logit, similarity):
        passage = feeder.ids_to_sent(pid)
        question = feeder.ids_to_sent(qid)
        print(passage)
        generated_question = feeder.decode_logit(logit)
        print('  reference: {}'.format(question))
        print('  generate:  {}'.format(generated_question))
        print('  similarity:{}'.format(sim))
    return loss


class Evaluator(TrainFeeder):
    def __init__(self, dataset=None):
        super(Evaluator, self).__init__(Dataset() if dataset is None else dataset)
        self.discriminator_feed = self.generator_feed = None


    def create_feed(self, answer, question):
        #question = question.split(' ')
        #answer = answer.split(' ')
        aids = self.sent_to_ids(answer)
        qids = self.sent_to_ids(question)
        qv, _ = self.label_vector(question)
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


    def evaluate_discriminator(self, sess, model):
        self.prepare('dev')
        if self.discriminator_feed is None:
            pids, qids, labels, kb = self.next()
            self.discriminator_feed = model.feed_discriminator(pids, qids, labels, kb)
        return evaluate_discriminator(sess, model, self, None, False, self.discriminator_feed)


    def evaluate_generator(self, sess, model):
        self.prepare('dev')
        if self.generator_feed is None:
            pids, _, _, kb = self.next()
            self.generator_feed = model.feed_generator(pids, kb)
        return evaluate_generator(sess, model, self, None, False, self.generator_feed)


if __name__ == '__main__':
    from model import Model
    import tensorflow as tf
    evaluator = Evaluator()
    model = Model(evaluator.dataset.qi2c, config.checkpoint_folder, False)
    with tf.Session() as sess:
        model.restore(sess)
        evaluator.prepare('dev')
        for question, answer in evaluator.data[:10]:
            evaluator.predict(sess, model, answer, question)