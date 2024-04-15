# RL_FlappyBird

## Overview

Using DQN to play Flappy Bird. Implemented by PyTorch.

Improvement: Fixed Q-Network

`RL.md`: Notes about DQN

## Dependencies:

- PyTorch 1.4.0
- pygame 1.9.6

## How to Run?

play the game with pretrained weights:

(set FPS to 480, achieve 2000 score in 10 minutes.)

```
python main.py --play
```

train:

```
python main.py
```

resume training with pretrained weights:

```
python main.py --resume --ckpt_path 'path'
```

## Disclaimer

This work is based on the following repos:

1. [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)
2. [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
3. [xmfbit/DQN-FlappyBird](https://github.com/xmfbit/DQN-FlappyBird)

