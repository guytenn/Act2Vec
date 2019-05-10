from typing import List
import os
import numpy as np
import ujson as json


class ExperimentLog:
    def __init__(self, num_iters, num_vals):
        self.num_iters = num_iters
        self.num_vals = num_vals
        self.means = []
        self.std = None
        self.current_observation_num = 0
        self.n = 0

    def observe(self, values: List):
        if len(self.means) == 0:
            raise ValueError('You must call new_trial before every new trial you run')
        for i, val in enumerate(values):
            self.means[self.n - 1][i][self.current_observation_num] = values[i]
        self.current_observation_num += 1

    def new_trial(self):
        self.means.append(np.zeros((self.num_vals, self.num_iters)))
        self.current_observation_num = 0
        self.n += 1

    def finalize(self):
        if self.current_observation_num == 0:
            self.current_observation_num = self.num_iters
        if self.n == 1:
            self.means = self.means[0][:, 0:self.current_observation_num+1]
        else:
            self.means = np.array(self.means)
            self.means = self.means[:, :, 0:self.current_observation_num+1]
            self.std = np.std(self.means, axis=0)

    def save(self, name, out_dir: str, unique_num: int = 0):
        out_prefix = out_dir

        if not os.path.exists(out_prefix):
            os.makedirs(out_prefix)
        filename = name + str(unique_num) + ".json"
        full_out_path = os.path.join(out_prefix, filename)

        self.finalize()
        data = {'means': self.means, 'std': self.std}
        with open(full_out_path, 'w') as outfile:
            json.dump(data, outfile)


if __name__ == "__main__":
    num_iters = 100
    num_vals = 3
    num_trials = 5

    logger = ExperimentLog(num_iters, num_vals)

    for trial in range(num_trials):
        logger.new_trial()
        rewards = np.random.randint(5, size=num_iters)
        entropy = np.random.randint(10, size=num_iters)
        learning_rate = np.random.randint(15, size=num_iters)
        for i in range(num_iters):
            logger.observe([rewards[i], entropy[i], learning_rate[i]])

    logger.save('test', '.')
