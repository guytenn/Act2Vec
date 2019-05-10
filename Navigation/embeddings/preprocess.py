from utils.utils import printProgress
import os
import ujson as json
from utils.corpus_utils import unbag_corpus


class NavigationDataLoader:
    def __init__(self, path, seq_len, max_replays=None, continuous=False, with_print=False):
        self.seq_len = seq_len

        actions = []
        replay_files = os.listdir(path)
        N = len(replay_files)
        if max_replays is None:
            max_replays = N
        i = 0
        self.n_replays = 0
        for fname in replay_files:
            if i >= max_replays:
                break
            if with_print:
                printProgress(i, N)
            file_path = os.path.join(path, fname)
            if not os.path.isdir(file_path):
                with open(file_path) as f:
                    replay_actions = json.load(f)
                actions.append(replay_actions)
                self.n_replays += 1
            i += 1
        for i in range(len(actions)):
            actions[i] = self.divide_into_composite(actions[i], continuous)
        self.actions = actions
        if with_print:
            print('Data loaded.')

    def divide_into_composite(self, actions: list, continuous=False):
        composite_actions = []
        ind = 0
        actions = ''.join(actions)
        while ind < len(actions):
            curr_word = actions[ind: ind + self.seq_len]
            if (len(curr_word) < seq_len):
                break
            composite_actions.append(curr_word)
            if continuous:
                ind += 1
            else:
                ind += self.seq_len
        return composite_actions


if __name__ == "__main__":
    replay_path = 'Navigation/action_corpus/player1/'
    embedding_dims = [3, 5, 10]
    seq_lens = [2, 3, 4]
    windows = [2, 3, 5]
    bag_of_words = True

    from gensim.test.utils import get_tmpfile
    from gensim.models import Word2Vec


    path_w2vec = get_tmpfile("act2vec.model")

    for seq_len in seq_lens:
        DL = NavigationDataLoader(replay_path, seq_len, continuous=True, with_print=True)
        orig_corpus = DL.actions
        for window in windows:
            for embedding_dim in embedding_dims:
                if not bag_of_words:
                    print('Unbagging corpus...')
                    corpus = unbag_corpus(orig_corpus, window=window)
                else:
                    corpus = orig_corpus

                print('Building Action2Vec model with window size {} and embedding dim {}'.format(window, embedding_dim))

                model = Word2Vec(corpus, size=embedding_dim, window=window, min_count=1, workers=4)

                vocab = list(model.wv.vocab)

                # print("Training word2vec model")
                model.train(corpus, total_examples=len(corpus), epochs=50)

                desc = "act2vec_dataset:navigation" + \
                       "_dim:" + str(embedding_dim) + \
                       "_win:" + str(window) + \
                       "_seq_len:" + str(seq_len)

                if bag_of_words:
                    desc += "_bagofwords"

                model.save("Navigation/embeddings/embeddings/" + desc + ".model")
