import QuickDraw.stable_baselines as stable_baselines
from QuickDraw.stable_baselines.common.vec_env import SubprocVecEnv
from QuickDraw.environments.rect_painter_gym import PainterGym
import tensorflow as tf
import os
import numpy as np
from gensim.models import Word2Vec
from utils.experiment_log import ExperimentLog
import argparse

parser = argparse.ArgumentParser(description='Parser for Rect Painter Trainer')
parser.add_argument('test', type=str, help='Name of test to execute')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--n_iters', type=int, default=int(10e6), help='Number of iterations')
parser.add_argument('--square_size', type=int, default=4, help='Number of sides to square')
parser.add_argument('--n_trials', type=int, default=1, help='Number of trials for test')
parser.add_argument('--save_id', type=int, default=0, help='id for log')
parser.add_argument('--embedding_dim', type=int, default=10, help='id for log')
parser.add_argument('--window', type=int, default=2, help='id for log')

args = parser.parse_args()

def action_translator(action):
    if action == 0:
        return 'R' * 20
    elif action == 1:
        return 'L' * 20
    elif action == 2:
        return 'U' * 20
    elif action == 3:
        return 'D' * 20
    elif action == 4:
        return 'L' * 10 + 'U' * 10
    elif action == 5:
        return 'U' * 10 + 'R' * 10
    elif action == 6:
        return 'R' * 10 + 'D' * 10
    elif action == 7:
        return 'D' * 10 + 'L' * 10
    elif action == 8:
        return 'R' * 10 + 'U' * 10
    elif action == 9:
        return 'U' * 10 + 'L' * 10
    elif action == 10:
        return 'L' * 10 + 'D' * 10
    elif action == 11:
        return 'D' * 10 + 'R' * 10
    elif action == 12:
        return 'NULL'

stroke_length = 20
embedding_dim = args.embedding_dim
window = args.window
dataset = "square"
n_cpu = 256
n_steps = args.square_size * 10
n_iters = args.n_iters
n_trials = args.n_trials
test = args.test
save_id = args.save_id

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

print('' * 100)
print('Runing test ' + test)

n_batch = n_steps * n_cpu
n_updates = n_iters // n_batch

desc = os.path.join("QuickDraw/embeddings/embeddings/", "act2vec_dataset:" + dataset + "_wordlen:" + str(stroke_length) +
                            "_dim:" + str(embedding_dim) + "_win:" + str(window))

act2vec_model = Word2Vec.load(desc + ".model")

print('Found {} words in model'.format(len(list(act2vec_model.wv.vocab))))

env = SubprocVecEnv([lambda: PainterGym(width=256, height=256, embd_dim=embedding_dim,
                                        action_translator=action_translator,
                                        square_size=args.square_size,
                                        act2vec_model=act2vec_model,
                                        action_buff_size=1,
                                        action_space='discrete',
                                        emb_type=test)
                     for i in range(n_cpu)])

logger = ExperimentLog(n_updates, 1)

for trial in range(n_trials):
    logger.new_trial()
    env.random_with_seed()

    model = stable_baselines.PPO2(policy="MlpPolicy", env=env,
                                  n_steps=n_steps,
                                  noptepochs=3,
                                  ent_coef=0.01,
                                  lam=0.95,
                                  cliprange=0.1,
                                  learning_rate=5e-4,
                                  verbose=2,
                                  tensorboard_log='tensorboard/log_tmp')

    print('-' * 100)
    print('Starting Trial {}, Test {}'.format(trial, test))
    model.learn(total_timesteps=n_iters, seed=0, exp_logger=logger)

    model.save("models/saves/rect_painter_ppo2_test-{}_trial{}".format(test, trial + 1))

logger.save('ppo_test_{}_square-size_{}_log'.format(test, args.square_size), 'logs/rect_painter', save_id)