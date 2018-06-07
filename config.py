checkpoint_folder = './checkpoint'
log_folder = './log'
embedding_dim = 128
encoder_hidden_dim = 128
num_passage_encoder_layers = 4
num_passage_residual_layers = 2
num_question_encoder_layers = 2
num_question_residual_layers = 1
decoder_hidden_dim = 128
dense_vector_dim = 1024
max_question_len = 50
batch_size=32
NULL = '<NULL>'
OOV = '<OOV>'
SOS = '<SOS>'
EOS = '<EOS>'
NULL_ID = 0
OOV_ID = 1
SOS_ID = 2
EOS_ID = 3
keep_prob = 0.8
raw_train_file = './data/zhidao.train.json'
raw_dev_file = './data/zhidao.dev.json'
raw_test_file = './data/zhidao.test.json'
train_file = './generate/train.txt'
dev_file = './generate/dev.txt'
test_file = './generate/test.json'
word_vocab_file = './generate/vocab.word.txt'
char_vocab_file = './generate/vocab.char.txt'
stopwords_file = './data/stopwords.txt'
answer_limit = 400
word_vocab_size = 100000
char_vocab_size = 100000