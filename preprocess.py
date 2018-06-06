import json
import os
import utils
import config
import re
from collections import defaultdict

stop_words = set(utils.read_all_lines(config.stopwords_file))

def create_vocab(filename):
    vocab = defaultdict(lambda: 0)
    for line in utils.read_all_lines(filename):
        for word in line.split(' '):
            vocab[word] += 1
    vocab = sorted(vocab.items(), key=lambda x:-x[1])
    utils.write_all_lines(config.vocab_file, ['{}:{}'.format(w,c) for w,c in vocab])


def prepare_dataset_with_document(source, target):
    passages = []
    for line in utils.read_all_lines(source):
        sample = json.loads(line)
        documents = sample['documents']
        questions = [sample['segmented_question']] + [doc['segmented_title'] for doc in documents]
        question_words = set(questions[0]) - stop_words
        questions = [' '.join(question) for question in questions]
        for doc in documents:
            for passage in doc['segmented_paragraphs']:
                passage_words = set(passage) - stop_words
                common = question_words & passage_words
                passage = rip_marks(' '.join(passage))
                if len(common) / len(question_words) > 0.3 and len(passage) > 2 * len(questions[0]):
                    passages.append(passage)
                    passages += questions
                    passages += '<P>'
    utils.write_all_lines(target, passages)


def rip_marks(text):
    r = re.sub(r'< ([A-Za-z0-9 /\"=]+) >', r'', text)
    r = re.sub(r'& [a-zA-Z]+ ;', r'', r)
    r = r.replace('　', ' ')
    r = r.replace('\t', ' ')
    r = re.sub(r'  +', r' ', r)
    r = r.strip()
    return r


if __name__ == '__main__':
    prepare_dataset_with_document(config.raw_train_file, config.train_file)
    prepare_dataset_with_document(config.raw_dev_file, config.dev_file)
    create_vocab(config.train_file)
