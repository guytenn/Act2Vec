import gym
from gym import spaces
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
# import visdom
from QuickDraw.embeddings.preprocess import PainterDataLoader
import datetime
import os
import re


class PainterGym(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, embd_dim,
                 act2vec_model, action_translator=None,
                 action_buff_size=50, square_size=4, action_space='discrete',
                 use_action_representation=True, emb_type='action2vec'):
        super(PainterGym, self).__init__()

        if action_space == 'discrete':
            self.action_space = spaces.Discrete(8)

        if emb_type == 'one_hot':
            embd_dim = 8

        if use_action_representation:
            self.observation_space = spaces.Box(low=-500, high=500,
                                                shape=(action_buff_size, embd_dim), dtype=np.float)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(height, width, 1), dtype=np.uint8)

        self.square_size = square_size
        self.width = width
        self.height = height
        self.num_actions = 8
        self.use_action_representation = use_action_representation
        self.canvas = None
        self.embedding_canvas = None
        self.action_buff_size = action_buff_size
        self.actions = None
        self.embd_dim = embd_dim
        self.action_translator = action_translator
        self.reward_network = None
        self.vocab = list(act2vec_model.wv.vocab)
        self.word2vec = act2vec_model
        self.emb_type = emb_type
        self.blank_embedding = np.zeros(embd_dim) #self.word2vec.wv['L'*len(self.vocab[0])] # self.word2vec.wv['S'*len(self.vocab[0])] #.np.zeros(embd_dim)
        self.num_saved_images = 0
        self.done = False
        self.rand_embeddings = None

        self.reset()

    def NumActions(self):
        return self.num_actions

    def IsRunning(self):
        return not self.done

    def random_with_seed(self, seed):
        np.random.seed(seed)
        self.rand_embeddings = np.random.rand(self.num_actions, self.embd_dim)

    def step(self, action):
        actions_str = self.action_translator(action)
        done_E = done_len = done_r = done_oob = False

        if 'E' in actions_str:
            done_E = True
        else:
            if actions_str is not 'NULL':
                if self.emb_type == 'one_hot':
                    action_emb = np.eye(self.embd_dim)[action]
                elif self.emb_type == 'random':
                    action_emb = self.rand_embeddings[action]
                elif self.emb_type == 'action2vec':
                    action_emb = self.word2vec.wv[actions_str]
                elif self.emb_type == 'action2vec_normalized':
                    action_emb = self.word2vec.wv[actions_str]
                    norm_emb = np.sqrt(np.sum(action_emb**2))
                    action_emb = action_emb / norm_emb
                else:
                    raise ValueError('Uknown type ' + self.emb_type)

                if self.action_buff_size == 1:
                    self.embedding_canvas[0, :] += action_emb
                else:
                    self.embedding_canvas[len(self.actions), :] = action_emb #/ 12.

                self.actions.append(actions_str)

            self.canvas, done_oob = PainterDataLoader.image_from_actions(self.actions, width=self.width, height=self.height)

            if len(self.actions) == self.square_size:
                done_len = True

        if done_len:
            r = self.get_reward()
            # print(self.actions)
        else:
            r = 0
        # r = self.get_reward()

        self.done = done_r or done_len or done_E # or done_oob


        if self.use_action_representation:
            state = self.embedding_canvas
        else:
            state = self.canvas

        return state, len(self.actions), r, self.done, {}

    def Act(self, action, frame_repeat, seq_len=1):
        _, _, r, _, _ = self.step(action)
        return r

    def Observation(self):
        if self.use_action_representation:
            return self.embedding_canvas
        else:
            return self.canvas

    def reset(self):
        self.done = False
        self.canvas = np.zeros((self.height, self.width, 1), dtype='float')
        self.embedding_canvas = np.ones((self.action_buff_size, self.embd_dim)) * self.blank_embedding
        self.actions = []
        if self.use_action_representation:
            state = self.embedding_canvas
        else:
            state = self.canvas
        return state, 0

    def render(self, mode='human', close=False):
        if self.canvas is None:
            return
        plt.figure(0)
        plt.imshow(self.canvas[:, :, 0])
        plt.show()

    def get_reward(self):
        actions_str = ''.join(self.actions)

        if len(re.findall('L*[!L]', actions_str)) != 1 or \
                len(re.findall('R*[!R]', actions_str)) != 1 or \
                len(re.findall('U*[!U]', actions_str)) != 1 or \
                len(re.findall('D*[!D]', actions_str)) != 1:
            return -0.1

        l_span = re.search('L*[!L]', actions_str).span(0)
        r_span = re.search('R*[!R]', actions_str).span(0)
        u_span = re.search('U*[!U]', actions_str).span(0)
        d_span = re.search('D*[!D]', actions_str).span(0)

        RULD = [r_span[0], u_span[0], l_span[0], d_span[0]]
        order = np.argsort(RULD)
        lengths = [r_span[1]-r_span[0], u_span[1]-u_span[0], l_span[1]-l_span[0], d_span[1]-d_span[0]]
        W = (self.square_size / 4) * len(self.vocab[0])
        length_rewards = [min(float(l), W) for l in lengths]

        if all(order == [0, 1, 2, 3]) or \
                all(order == [1, 2, 3, 0]) or \
                all(order == [2, 3, 0, 1]) or \
                all(order == [3, 0, 1, 2]) or \
                all(order == [3, 2, 1, 0]) or \
                all(order == [0, 3, 2, 1]) or \
                all(order == [1, 0, 3, 2]) or \
                all(order == [2, 1, 0, 3]):
            return sum(length_rewards) / len(actions_str)

        return 0








