import numpy as np
import sys

def printProgress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    :param iteration: current iteration
    :param total: total iterations
    :param prefix:  prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration >= total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def add_mean_cluster_embeddings(X, vocab, actions: list):

    action_dict = dict.fromkeys(actions)
    for action in action_dict:
        action_dict[action] = [np.zeros(X.shape[1]), 0]

    for action in actions:
        for ww, word in enumerate(vocab):
            if action in word:
                action_dict[action][0] += X[ww]
                action_dict[action][1] += 1
    X_new = np.zeros((X.shape[0] + len(actions), X.shape[1]))
    X_new[0: X.shape[0]] = X
    vocab_new = vocab
    for i, key in enumerate(action_dict):
        X_new[X.shape[0] + i] = action_dict[key][0] / action_dict[key][1]
        vocab_new += [key]
    return X_new, vocab_new
