from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ujson as json
from StarCraft2.embeddings.preprocess import StarCraftDataLoader

save_all = False  # when set to true will fetch all files ending with n_replays and save to different plots
embedding_dim = 5
window = 3
save_fig = True
annotate = True
races = ['terran', 'zerg', 'protoss', 'common']
ignore_null = True

fname = "act2vec_dataset:starcraft" + \
       "_dim:" + str(embedding_dim) + \
       "_win:" + str(window)

if ignore_null:
    fname += "_ignorenull"

fname += ".model"

fnames = [fname]

for fname in fnames:
    model = Word2Vec.load("StarCraft2/embeddings/embeddings/" + fname)

    vocab = list(model.wv.vocab)
    vocab_verbose = [''] * len(vocab)

    X = model[vocab]
    N = len(X)
    print('Vocabulary is of size {}'.format(N))

    if save_fig:
        with open('StarCraft2/embeddings/valid_actions.json') as f:
            mapping = json.load(f)
        for i, key in enumerate(vocab):
            vocab_verbose[i] = key + ':' + mapping[key]

    # Plot embedding of all possible actions
    print("Projecting embeddings")
    reduced_model = TSNE(n_components=2)
    X_reduced = reduced_model.fit_transform(X)
    df_global = pd.DataFrame(X_reduced, index=vocab, columns=['x', 'y'])

    # Divide according to race
    df = dict()
    for race in races:
        with open('StarCraft2/embeddings/{}_actions.json'.format(race)) as f:
            race_actions = json.load(f)
            df[race] = dict()
            df[race]['x'] = [df_global['x'][i] for i in range(N) if vocab[i] in race_actions]
            df[race]['y'] = [df_global['y'][i] for i in range(N) if vocab[i] in race_actions]


    if save_fig:
        if annotate:
            fig = plt.figure(figsize=(70, 70))
        else:
            fig = plt.figure(figsize=(7, 7))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['b', 'r', 'g', 'k']
    for c, race in enumerate(races):
        ax.scatter(df[race]['x'], df[race]['y'], color=colors[c], s=2000)

    if save_fig:
        if len(races) == 1:
            save_path = 'StarCraft2/embeddings/' + races[0]
        else:
            save_path = 'StarCraft2/embeddings/'
        plt.savefig(os.path.join(save_path, 'embeddings_{}.pdf'.format(fname[9:].split('.')[0])), format='pdf')
        if annotate:
            for i, txt in enumerate(vocab_verbose):
                if len(races) > 1 or vocab[i] in race_actions:
                    ax.annotate(txt, (df_global['x'][i], df_global['y'][i]))
            plt.savefig(os.path.join(save_path, 'embeddings_annotated_{}.pdf'.format(fname[9:].split('.')[0])), format='pdf')
    else:
        plt.show()

