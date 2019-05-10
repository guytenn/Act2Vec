from utils.utils import printProgress
import os
import csv
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec


class StarCraftDataLoader:
    def __init__(self, path, ignore_null=True, max_replays=None, with_print=False):
        actions = []
        replay_files = os.listdir(path)
        N = len(replay_files)
        if max_replays is None:
            max_replays = N
        i = 0
        self.n_actions = 0
        self.n_replays = 0
        for fname in replay_files:
            if i >= max_replays:
                break
            if with_print:
                printProgress(i, N)
            file_path = os.path.join(path, fname)
            if not os.path.isdir(file_path):
                with open(file_path, 'r') as f:
                    reader = csv.reader(f, delimiter=' ')
                    replay_actions = []
                    for row in reader:
                        if row[0].isdigit():
                            if ignore_null and int(row[0]) == 0:
                                continue
                            replay_actions += row
                    actions.append(replay_actions)
                    self.n_replays += 1
                    self.n_actions += len(replay_actions)
            i += 1
        self.actions = actions
        if with_print:
            print('Data loaded.')
            print('Corpus size: {} actions, {} replays'.format(self.n_actions, self.n_replays))


if __name__ == "__main__":
    replay_path = 'StarCraft2/corpus_sample'
    ignore_null = True
    embedding_dims = [5, 10, 25, 50]
    windows = [2, 3, 5, 8, 10]

    DL = StarCraftDataLoader(replay_path, ignore_null=True, with_print=True)

    corpus = DL.actions
    path_w2vec = get_tmpfile("act2vec.model")

    for window in windows:
        for embedding_dim in embedding_dims:

            print('Building Action2Vec model with window size {} and embedding dim {}'.format(window, embedding_dim))

            model = Word2Vec(corpus, size=embedding_dim, window=window, min_count=1, workers=4)

            vocab = list(model.wv.vocab)

            print('Training model...')
            model.train(corpus, total_examples=len(corpus), epochs=50)

            desc = "test_act2vec_dataset:starcraft" + \
                   "_dim:" + str(embedding_dim) + \
                   "_win:" + str(window)

            if ignore_null:
                desc += '_ignorenull'

            print('Saving model...')

            model.save("Starcraft2/embeddings/embeddings/" + desc + ".model")
