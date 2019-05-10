from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ujson as json
from navigation.preprocess import LabNavigationDataLoader

save_all = False # when set to true will fetch all files ending with n_replays and save to different plots
embedding_dim = 3
seq_len = 3
window = 2
reduce_func = PCA
continuous = True
save_fig = False

UP = '↑'
LEFT = '←'
RIGHT = '→'
def map(action_str):
    new_str = ''
    for a in action_str:
        if a == '0':
            new_str += LEFT
        elif a == '1':
            new_str += RIGHT
        else:
            new_str += UP
    return new_str

if save_all:
    fnames = os.listdir('navigation/embeddings/lab')
else:
    fname = "word2vec_dataset:lab_apples" + \
            "_dim:" + str(embedding_dim) + \
            "_win:" + str(window) + \
            "_seq_len:" + str(seq_len)

    fnames = [fname]

for fname in fnames:
    model = Word2Vec.load("navigation/embeddings/lab/" + fname + ".model")

    vocab = list(model.wv.vocab)
    vocab = [word for word in vocab if ('WL-' not in word and 'WR-' not in word)]
    X = model[vocab]
    N = len(X)
    print('Vocabulary is of size {}'.format(N))

    # Plot embedding of all possible actions
    print("Projecting embeddings")
    reduced_model = reduce_func(n_components=2)
    X_reduced = reduced_model.fit_transform(X)
    df = pd.DataFrame(X_reduced, index=vocab, columns=['x', 'y'])

    if save_fig:
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'], color='b', marker='.')

    for i, txt in enumerate(vocab):
        ax.annotate(map(txt), (df['x'][i], df['y'][i]))

    if save_fig:
        plt.savefig('navigation/images/embeddings_dim_{}_seq_len_{}_win_{}.pdf'.
                    format(embedding_dim, seq_len, window), format='pdf')
    else:
        plt.show()

