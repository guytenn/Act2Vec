from QuickDraw.preprocess import PainterDataLoader
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

embedding_dim = 5
stroke_length = 20
window = 2
dataset = "square"
UP = '↑'
LEFT = '←'
RIGHT = '→'
DOWN = '↓'

desc = "word2vec_dataset:" + dataset + "_wordlen:" + str(stroke_length) + \
       "_dim:" + str(embedding_dim) + "_win:" + str(window)

model = Word2Vec.load("QuickDraw/embeddings/" + desc + ".model")

strokes = ['L' * 20, 'R' * 20, 'U' * 20, 'D' * 20,
           'L' * 10 + 'U' * 10, 'U' * 10 + 'L' * 10,
           'R' * 10 + 'U' * 10, 'U' * 10 + 'R' * 10,
           'L' * 10 + 'D' * 10, 'D' * 10 + 'L' * 10,
           'R' * 10 + 'D' * 10, 'D' * 10 + 'R' * 10]

annotations = [LEFT + LEFT, RIGHT + RIGHT, UP + UP, DOWN + DOWN,
               LEFT + UP, UP + LEFT,
               RIGHT + UP, UP + RIGHT,
               LEFT + DOWN, DOWN + LEFT,
               RIGHT + DOWN, DOWN + RIGHT]

X = model[strokes]
N = len(X)

print("Projecting strokes")
reduced_model = PCA(n_components=2)
X_kmeans_reduced = reduced_model.fit_transform(X)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
df = pd.DataFrame(X_kmeans_reduced, index=[str(i) for i in range(N)], columns=['x', 'y'])
ax.scatter(df['x'], df['y'], color='b')
plt.savefig('QuickDraw/square_embeddings.pdf', format='pdf')

for i in range(N):
    ax.annotate(annotations[i], (df['x'][i] - 0.1 - np.random.rand(), df['y'][i] + 0.1 + np.random.rand()))

plt.savefig('QuickDraw/square_embeddings_annotated.pdf', format='pdf')

print('done')

