__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'


import numpy as np
import tensorflow as tf
import time
import jellyfish


DEBUG_FLAG = True


def embed(X, n_dim, n_iter):

    # checking
    if DEBUG_FLAG:
        print 'X[0,0]', X[0, 0]
        print 'X[1,1]', X[1, 1]

        print 'min X', np.min(X)
        print 'max X', np.max(X)

    # building model

    graph = tf.Graph()

    with graph.as_default():

        global_step = tf.Variable(0, trainable=False)

        name_embeddings = tf.Variable(tf.random_uniform([X.shape[0], n_dim], -2.0/np.sqrt(n_dim), 2.0/np.sqrt(n_dim)),
                                      name='name_embeddings')

        X_var = tf.constant(X)

        mt = tf.matmul(name_embeddings, name_embeddings, transpose_b=True, name='mt')
        loss = tf.nn.l2_loss((mt - X_var), name='loss') * 2 / (X.shape[0] ** 2)

        learning_rate = tf.train.exponential_decay(0.1, global_step, 100000, 0.96, staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True,
                                                       allow_soft_placement=True)) as session:

        tf.global_variables_initializer().run()

        for i in xrange(n_iter*1000):

            start_time = time.time()
            _, loss_val = session.run([optimizer, loss])
            if DEBUG_FLAG and np.random.rand() < 0.01:
                print i, 'f_loss_val', loss_val, time.time() - start_time

        name_embeddings = name_embeddings.eval()

    return name_embeddings


def embed_names(names, n_dim, n_iter=100):
    X_name = np.eye(len(names), dtype=np.float32)
    for i in xrange(0, len(names)):
        for j in xrange(i + 1, len(names)):
            dis = 2*jellyfish.jaro_winkler(names[i], names[j])-1
            X_name[i, j] += dis
            X_name[j, i] += dis

    return embed(X_name, n_dim, n_iter)


if __name__ == '__main__':
    names = [
        u'Amy Tan',
        u'Desmond',
        u'C L',
        u'Joey Lim',
        u'Nicole Tan',
        u'Desmond Ng',
        u'Cindy Lim',
        u'Joey L',
    ]

    name_embeddings = embed_names(names, 5, 10)



