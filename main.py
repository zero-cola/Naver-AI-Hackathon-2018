# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

np.random.seed(777)
tf.set_random_seed(1)

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output_score, feed_dict={x: preprocessed_data, output_keep_prob: 1.0,
                                                 phase: 0})

        point = pred.squeeze(axis=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1, balance=False, verbose=True):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)

    iter_review = np.array([np.array(review) for review, _ in iterable])
    iter_label = np.array([int(label) for _, label in iterable])


    p = np.random.permutation(length)

    iter_review, iter_label = iter_review[p], iter_label[p]

    if verbose:
        score_sum = sum(iter_label)
        print('score_mean = ', score_sum/length)

        # Label별 개수 출력
        print(np.bincount(iter_label)[1:])

    min_class_freq = np.min(np.bincount(iter_label)[1:])
    # if verbose:
    #     print('min_class_freq :', min_class_freq)
    # min_class_freq = 100000

    if balance:
        iter_dict = dict.fromkeys(range(1,11))
        for label in iter_dict:
            iter_dict[label] = []

        for i in range(length):
            iter_dict[int(iter_label[i])].append(iter_review[i])

        new_iter_review = []
        new_iter_label = []
        for label, reviews in iter_dict.items():
            new_iter_review[len(new_iter_review):] = reviews[:min_class_freq]
            new_iter_label[len(new_iter_label):] = [label] * len(reviews[:min_class_freq])

        new_iter_review = np.array(new_iter_review)
        new_iter_label = np.array(new_iter_label)

        length = len(new_iter_label)

        p = np.random.permutation(length)
        new_iter_review, new_iter_label = new_iter_review[p], new_iter_label[p]


    for n_idx in range(0, length, n):
        if not balance:
            yield iter_review[n_idx:min(n_idx + n, length)], iter_label[n_idx:min(n_idx + n, length)]
        if balance:
            yield new_iter_review[n_idx:min(n_idx + n, length)], new_iter_label[n_idx:min(n_idx + n, length)]



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=30)
    args.add_argument('--batch', type=int, default=500)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding', type=int, default=251)
    args.add_argument('--validate', type=float, default=0)
    args.add_argument('--dropout', type=float, default=0.49)
    args.add_argument('--lr', type=float, default=0.0004)
    args.add_argument('--lrdecay', type=float, default=0.77)
    args.add_argument('--min_lr', type=float, default=0.0001)
    args.add_argument('--l2_reg', type=float, default=0.01)
    args.add_argument('--balance', default=False, action='store_true')


    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    hidden_layer_size = 1280
    # learning_rate = config.lr
    character_size = 251
    filter_sizes = [2, 3, 4, 5, 6]
    sec_filter_sizes = [1, 2, 3]
    num_filters = 256

    # dropout keep prob
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
    phase = tf.placeholder(tf.bool, name='phase') # train: 1, test: 0

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)

    # LSTM
    hidden_size = 251
    with tf.name_scope("bidirectional-lstm"):
        rnn_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        rnn_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

        (output_fw, output_bw), last_state = \
            tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, embedded, dtype=tf.float32)

        lstm_outputs = tf.concat([output_fw, output_bw], axis=2)

        print(lstm_outputs)
        embedded_expanded = tf.expand_dims(lstm_outputs, -1)
        print(embedded_expanded)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):

            # Convolution Layer
            conv = tf.layers.conv2d(embedded_expanded,
                                    filters=num_filters,
                                    kernel_size=[filter_size, hidden_size*2],
                                    strides=[1, 1],
                                    activation=tf.nn.relu,)

            # Maxpooling Layer
            pooled = tf.layers.max_pooling2d(conv,
                                    [3, 1],
                                    strides=[2, 1],)
            pooled = tf.transpose(pooled, [0, 1, 3, 2])

            for j, sec_filter_size in enumerate(sec_filter_sizes):
                with tf.name_scope("conv2-maxpool-%s" % sec_filter_size):
                    # Convolution Layer
                    conv2 = tf.layers.conv2d(pooled,
                                            filters=num_filters,
                                            kernel_size=[sec_filter_size, pooled.shape[2]],
                                            strides=[1, 1],
                                            activation=tf.nn.relu, )
                    # Maxpooling Layer
                    pooled2 = tf.layers.max_pooling2d(conv2,
                                                     [conv2.shape[1], 1],
                                                     strides=[1, 1], )
                    pooled_outputs.append(pooled2)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, h_pool.shape[-1]])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, output_keep_prob)

    fc_1 = tf.contrib.layers.fully_connected(h_drop, int(hidden_layer_size),
                                             activation_fn=tf.nn.relu,)
    fc_1 = tf.nn.dropout(fc_1, output_keep_prob)                                         
    output = tf.contrib.layers.fully_connected(fc_1, 1, activation_fn=None)

    # 예측 점수
    output_tanh = tf.tanh(output)
    output_score = (output_tanh + 1) * 4.5 + 1

    # loss와 optimizer
    mean_square_error = tf.reduce_mean(tf.square(output_score - y_))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_square_error)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)

        if config.validate > 0:

            validate_len = int(dataset_len * config.validate)

            ds_review = np.array([np.array(review) for review, _ in dataset])
            ds_label = np.array([int(label) for _, label in dataset])

            global_perm = np.random.RandomState(seed=777).permutation(dataset_len)
            ds_review, ds_label = ds_review[global_perm], ds_label[global_perm]

            dataset = MovieReviewDataset(remake=True,
                                         new_review=ds_review[validate_len:],
                                         new_labels=ds_label[validate_len:])
            dataset_len = len(dataset)
            validate_dataset = MovieReviewDataset(remake=True,
                                         new_review=ds_review[:validate_len],
                                         new_labels=ds_label[:validate_len])
            validate_dataset_len = len(validate_dataset)


        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        # epoch마다 학습을 수행합니다.

        local_lr = config.lr
        for epoch in range(config.epochs):

            print('='*20)
            avg_loss = 0.0
            train_score_dict = {1: list(), 2: list(), 3: list(), 4: list(), 5: list(),
                          6: list(), 7: list(), 8: list(), 9: list(), 10: list()}
            val_score_dict = {1: list(), 2: list(), 3: list(), 4: list(), 5: list(),
                          6: list(), 7: list(), 8: list(), 9: list(), 10: list()}

            print('training...')
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch, config.balance)):
                labels = np.array(labels).reshape(-1, 1)
                _, loss, y_pred= sess.run([train_step, mean_square_error, output_score],
                                         feed_dict={x: data, y_: labels,
                                                    learning_rate: local_lr,
                                                    output_keep_prob: config.dropout,
                                                    phase: 1})

                print('Batch : ', i + 1, '/', one_batch_size,
                      ', MSE in this minibatch: ', float(loss))


                for _y_pred, _y_real in zip(y_pred, labels):
                    train_score_dict[int(_y_real)].append(_y_pred[0])

                avg_loss += float(loss)

            if local_lr * config.lrdecay < config.min_lr:
                config.lrdecay = 1.0

            local_lr = local_lr * config.lrdecay

            print('in training set...')
            for score in train_score_dict:
                print('label:{:2}'.format(score), 'predict_mean: {:.3f}'
                      .format(sum(train_score_dict[score])/len(train_score_dict[score])))


            if config.validate > 0:
                validate_avg_loss = 0.0
                print('validating...{} data..'.format(len(validate_dataset)))
                for i, (data, labels) in enumerate(
                        _batch_loader(validate_dataset, config.batch, balance=False, verbose=True)):
                    labels = np.array(labels).reshape(-1, 1)
                    loss, y_pred = sess.run([mean_square_error, output_score],
                                          feed_dict={x: data, y_: labels, output_keep_prob: 1.0,
                                                     phase: 0})

                    # print(info)
                    validate_avg_loss += float(loss)

                    for _y_pred, _y_real in zip(y_pred, labels):
                        val_score_dict[int(_y_real)].append(_y_pred[0])

                one_val_batch_size = validate_dataset_len // config.batch
                if validate_dataset_len % config.batch != 0:
                    one_val_batch_size += 1
                print('in validation set...')
                for score in val_score_dict:
                    print('label:{:2}'.format(score), 'predict_mean: {:.3f}'
                          .format(sum(val_score_dict[score]) / len(val_score_dict[score])))

            print('==> epoch:', epoch, ' train_loss:', float(avg_loss / one_batch_size))
            if config.validate > 0:
                print('validate_loss: ', float(validate_avg_loss / one_val_batch_size))
                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                            train__loss=float(avg_loss / one_batch_size), step=epoch,
                            val__loss=float(validate_avg_loss / one_val_batch_size))
            else:
                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                            train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
