import tensorflow as tf
import config
import utils
import numpy as np
from evaluator import Evaluator, evaluate_discriminator
from data import TrainFeeder, Dataset
from model import Model


def diagm(name, value):
    small = np.min(value)
    big = np.max(value)
    assert np.all(np.isfinite(value)), '{} contains invalid number'.format(name)
    print('{}: {:>.4f} ~ {:>.4f}'.format(name, small, big))


def run_discriminator_epoch(itr, sess, model, feeder, evaluator, writer):
    feeder.prepare('train')
    nbatch = 0
    while not feeder.eof():
        loss = evaluate_discriminator(sess, model, feeder, writer, True)
        print('-----ITERATION {}, {}/{}, loss: {:>.4F}'.format(itr, feeder.cursor, feeder.size, loss))
        nbatch += 1
        if nbatch % 10 == 0:
            loss = evaluator.evaluate_discriminator(sess, model)
            print('===================DEV loss: {:>.4F}==============='.format(loss))
            model.save(sess)


def run_generator_epoch(itr, sess, model, feeder, evaluator, writer):
    feeder.prepare('train')
    nbatch = 0
    while not feeder.eof():
        pids, qids, _, kb = feeder.next()
        feed = model.feed_generator(pids, kb)
        summary, global_step, _, loss, similarity, question_logit = sess.run(
            [
                model.summary, model.global_step, model.generator_optimizer, model.generator_loss,
                model.norm_similarity, model.generator.question_logit
            ], feed_dict=feed)
        writer.add_summary(summary, global_step=global_step)
        pid, qid, sim = pids[0], qids[0][0], similarity[0]
        passage = feeder.ids_to_sent(pid)
        question = feeder.ids_to_sent(qid)
        print(passage)
        generated_question = feeder.decode_logit(question_logit[0])
        print('--------------------------------------')
        print('reference: {}'.format(question))
        print('generate:  {}'.format(generated_question))
        print('similarity:{}'.format(sim))
        print('-----ITERATION {}, {}/{}, loss: {:>.4F}'.format(itr, feeder.cursor, feeder.size, loss))
        nbatch += 1
        if nbatch % 10 == 0:
            loss = evaluator.evaluate_generator(sess, model)
            print('===================DEV loss: {:>.4F}==============='.format(loss))
            model.save(sess)


def train(auto_stop):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    evaluator = Evaluator(dataset)
    model = Model(dataset.ci2n, config.checkpoint_folder)
    with tf.Session() as sess:
        model.restore(sess)
        #utils.rmdir(config.log_folder)
        writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        model.summarize()
        itr = 0
        print('start training...')
        while True:
            itr += 1
            run_discriminator_epoch(itr, sess, model, feeder, evaluator, writer)
            #run_generator_epoch(itr, sess, model, feeder, evaluator, writer)


train(False)
