def unbag_corpus(corpus, window=1):
    new_corpus = []
    for line in corpus:
        for i in range(window, len(line) - window):
            words_left = ['WL-' + w for w in line[i-window: i]]
            words_right = ['WR-' + w for w in line[i+1: i+1+window]]
            new_corpus.append(words_left + [line[i]] + words_right)
    return new_corpus

if __name__ == '__main__':
    corpus = [['hello', 'my', 'friend', 'how', 'do', 'you', 'do'],
              ['i', 'am', 'fine', 'thank', 'you', 'how', 'are', 'you']]
    new_corpus = unbag_corpus(corpus, window=1)
    print(new_corpus)