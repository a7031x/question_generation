import tensorflow as tf
import config
import utils
import numpy as np
from evaluator import Evaluator
from data import TrainFeeder, Dataset
from model import Model


def diagm(name, value):
    small = np.min(value)
    big = np.max(value)
    assert np.all(np.isfinite(value)), '{} contains invalid number'.format(name)
    print('{}: {:>.4f} ~ {:>.4f}'.format(name, small, big))


def run_epoch(itr, sess, model, feeder, evaluator, writer):
    feeder.prepare('train')
    nbatch = 0
    while not feeder.eof():
        pids, qids, labels, kb = feeder.next()
        feed = model.feed(pids, qids, labels, kb)
        summary, global_step, _, loss, similarity = sess.run(
            [
                model.summary, model.global_step, model.optimizer, model.loss,
                model.similarity
            ], feed_dict=feed)
        writer.add_summary(summary, global_step=global_step)
        pid, qid, sim, lab = pids[0], qids[0], similarity[:len(qids[0])], labels[0]
        passage = feeder.ids_to_sent(pid)
        questions = [feeder.ids_to_sent(q) for q in qid]
        print(passage)
        for q,s,l in zip(questions, sim, lab):
            if q:
                print(' {} {:>.2F}: {}'.format(l, s, q))
        print('-----ITERATION {}, {}/{}, loss: {:>.4F}'.format(itr, feeder.cursor, feeder.size, loss))
        nbatch += 1
        if nbatch % 10 == 0:
            loss = evaluator.evaluate(sess, model)
            print('===================DEV loss: {:>.4F}==============='.format(loss))
            model.save(sess)


def train(auto_stop):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    evaluator = Evaluator(dataset)
    model = Model(len(dataset.chars), config.checkpoint_folder)
    with tf.Session() as sess:
        model.restore(sess)
        #utils.rmdir(config.log_folder)
        writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        model.summarize(writer)
        itr = 0
        print('start training...')
        while True:
            itr += 1
            run_epoch(itr, sess, model, feeder, evaluator, writer)


train(False)
