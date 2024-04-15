import sys
# sys.path.append("game/")
import game.wrapped_flappy_bird as game
# from BrainDQN import *
import collections
import shutil
import numpy as np
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim

import PIL.Image as Image

IMAGE_SIZE = (72, 128)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter( "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s" )

    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class ReplayMemory(object):
    empty_frame = np.zeros((128, 72), dtype=np.float32)
    empty_state = np.stack((empty_frame, empty_frame, empty_frame, empty_frame), axis=0)

    def __init__(self, max_size, options):
        self.buffer = collections.deque(maxlen=max_size)
        self.current_state = ReplayMemory.empty_state
        self.options = options

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        obs_batch = torch.tensor(np.array(obs_batch).astype('float32'))
        action_batch = torch.tensor(np.array(action_batch).astype('float32'))
        reward_batch = torch.tensor(np.array(reward_batch).astype('float32'))
        next_obs_batch = torch.tensor(np.array(next_obs_batch).astype('float32'))
        done_batch = torch.tensor(np.array(done_batch).astype('float32'))

        if self.options.cuda:
            obs_batch = obs_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_obs_batch = next_obs_batch.cuda()
            done_batch = done_batch.cuda()
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def store_state(self, o_next):
        self.current_state = np.append(self.current_state[1:, :, :], o_next.reshape((1,) + o_next.shape), axis=0)

    def append(self, o_next, action, reward, terminal):
        next_state = np.append(self.current_state[1:, :, :], o_next.reshape((1,) + o_next.shape), axis=0)
        self.buffer.append((self.current_state, action, reward, next_state, terminal))

        if not terminal:
            self.current_state = next_state
        else:
            self.current_state = ReplayMemory.empty_state

    def reset(self):
        self.current_state = ReplayMemory.empty_state

    def __len__(self):
        return len(self.buffer)


def preprocess(frame):
    """Do preprocessing: resize and binarize.

       Downsampling to 128x72 size and convert to grayscale
       frame -- input frame, rgb image with 512x288 size
    """
    im = Image.fromarray(frame).resize(IMAGE_SIZE).convert(mode='L')
    out = np.asarray(im).astype(np.float32)
    out[out <= 1.] = 0.0
    out[out > 1.] = 1.0
    return out


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint model to disk

        state -- checkpoint state: model weight and other info
                 binding by user
        is_best -- if the checkpoint is the best. If it is, then
                   save as a best model
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(filename, model):
    """Load previous checkpoint model

       filename -- model file name
       model -- DQN model
    """
    try:
        checkpoint = torch.load(filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    print('pretrained episode={}, epsilon={}'.format(episode, epsilon))
    model.load_state_dict(checkpoint['state_dict'])
    return episode, epsilon
