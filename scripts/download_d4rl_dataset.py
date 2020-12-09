import gym
import numpy as np

import argparse
import pickle
import sys


"""
Download D4RL dataset and store in lifelong_rl format
- Note this script requires having D4RL installed
- See: https://github.com/rail-berkeley/d4rl
"""

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str,
                    help='Which task to download dataset of (ex. halfcheetah-random-v0)')
parser.add_argument('--output', type=str, default='$$$',
                    help='What to name the output file of transitions (default: same as task)')
args = parser.parse_args(sys.argv[1:])

print('Getting dataset for %s' % args.task)

env = gym.make(args.task)
dataset = env.get_dataset()
dataset_len = len(dataset['observations'])

print('%d transitions found with average reward %.4f' % (dataset_len, dataset['rewards'].mean()))

# Note we store data as (obs, act, r, d, next_obs)
np_dataset = np.concatenate([
    dataset['observations'][:dataset_len-1],
    dataset['actions'][:dataset_len-1],
    dataset['rewards'][:dataset_len-1].reshape(dataset_len-1, 1),
    dataset['terminals'][:dataset_len-1].reshape(dataset_len-1, 1),
    dataset['observations'][1:],
], axis=-1)

output_file = args.output
if output_file == '$$$':
    output_file = args.task

with open('agent_data/%s.pkl' % output_file, 'wb') as f:
    pickle.dump(np_dataset, f)

print('Stored output in agent_data/%s.pkl' % output_file)
