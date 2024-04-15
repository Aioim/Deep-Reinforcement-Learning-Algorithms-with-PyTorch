




import game.wrapped_flappy_bird as game
# from BrainDQN import *
import shutil
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
# import PIL.Image as Image
from PIL import Image

from model import QNetwork
from DQNet import DQN
from utils import *

def train_dqn(options):
    max_episode = options.max_episode

    flappyBird = game.GameState()
    print(f'FPS {flappyBird.FPS}')

    rpm = ReplayMemory(options.rpm_size, options)  # DQN的经验回放池

    model = QNetwork()
    if options.resume and options.ckpt_path is not None:
        print ('load previous model weight: {}'.format(options.ckpt_path))
        episode, epsilon = load_checkpoint(options.ckpt_path, model)
    else:
        epsilon = options.init_e
        episode = 0

    if options.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=options.lr)
    algorithm = DQN(model, optimizer, epsilon, options)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < options.rpm_size/4:
        run_episode(algorithm, flappyBird, rpm, options)

    print(f'observation done {len(rpm)}')

    # 开始训练
    logname = time.strftime('%Y-%m-%d %M-%I-%S' , time.localtime())
    logger = get_logger(f'log/{logname}.log')
    best_reward = 0
    max_score = 0
    begin = time.time()
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part

        reward, loss, score = run_episode(algorithm, flappyBird, rpm, options)
        algorithm.epsilon = max(algorithm.final_e, algorithm.epsilon - algorithm.e_decrement)
        episode += 1
        max_score = max(max_score, score)

        if (episode)%10 == 0:
            logger.info('episode:[{}/{}]\tscore:{:.3f}\ttrain_reward:{:.5f}\tloss:{:.5f}'.format(
                episode, max_episode, score, reward, loss))
        
        # test part
        if (episode)%options.evaluate_freq == 0:
            eval_reward, score = evaluate(flappyBird, algorithm, options)
            mid = time.time()
            elapsed = round(mid-begin)
            logger.info('episode:[{}/{}]\tscore:{:.3f}\tepsilon:{:.5f}\ttest_reward:{:.5f}\t{}:{}'.format(
                episode, max_episode, score, algorithm.epsilon, eval_reward, elapsed//60, elapsed%60))
            if eval_reward > best_reward:
                save_path = f'ckpt/best_{score}.ckpt'
                save_checkpoint({
                    'episode': episode,
                    'epsilon': algorithm.epsilon,
                    'state_dict': model.state_dict(),
                    }, False, save_path
                )

        if (episode)%1000 == 0:
            save_path = f'ckpt/episode_{episode}.ckpt'
            save_checkpoint({
                'episode': episode,
                'epsilon': algorithm.epsilon,
                'state_dict': model.state_dict(),
                }, False, save_path
            )

    # 训练结束，保存模型
    save_path = f'ckpt/final_{episode}_{score}.ckpt'
    save_checkpoint({
        'episode': episode,
        'epsilon': algorithm.epsilon,
        'state_dict': model.state_dict(),
        }, False, save_path)

    mid = time.time()
    elapsed = round(mid-begin)
    logger.info('training completed, {} episiode, {}m {}s'.format(max_episode, elapsed//60, elapsed%60))
    print(f'max_score {max_score}')

def run_episode(model, flappyBird, rpm, options):
    """Train DQN

       model -- DQN model
       flappyBird -- environment
       rpm -- replay memory
       options -- resume previous model
    """

    time_step = 0
    total_reward = 0
    model.set_train()
    rpm.reset()
    flappyBird.reset()
    
    action = [1, 0]
    o, r, terminal = flappyBird.frame_step(action)
    o = preprocess(o)
    rpm.store_state(o)
    # rpm.append(o, action, r, terminal)
    score = 0
    loss = 0

    while True:
        obs = torch.tensor(rpm.current_state).unsqueeze(0)
        if options.cuda:
            obs = obs.cuda()
        action = model.get_action(obs)
        # adjust model.epsilon?
        score = max(score, flappyBird.score)
        o_next, r, terminal = flappyBird.frame_step(action)
        total_reward += options.gamma**time_step * r
        o_next = preprocess(o_next)
        rpm.append(o_next, action, r, terminal)
        
        if time_step % options.learning_freq == 0 and len(rpm) > options.rpm_size/4:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = rpm.sample(options.batch_size)
            
            loss = model.learn(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            model.global_step += 1
            if model.global_step % options.update_target_steps == 0:
                model.global_step = 0
                model.sync_weight()
        
        time_step += 1

        if terminal or score > options.max_score:
            break
    
    return total_reward, loss, score

def evaluate(flappyBird, model, options):
    """Test the behavor of dqn when training

       model -- dqn model
       episode -- current training episode
    """
    rpm = ReplayMemory(1, options)
    model.set_eval()
    rewards = []
    scores = []
    for _ in range(1):
        flappyBird.reset()
        action = [1, 0]
        o, r, terminal = flappyBird.frame_step(action)
        o = preprocess(o)
        rpm.append(o, action, r, terminal)
        time_step, total_reward = 0, 0
        score = 0

        while True:
            prev_o, a, r, o, terminal = rpm.sample(1)
            total_reward += options.gamma**time_step * r
            action = model.get_optim_action(o)
            score = max(score, flappyBird.score)
            o, r, terminal = flappyBird.frame_step(action)
            if terminal or score > options.max_score:
                break
            o = preprocess(o)
            rpm.append(o, action, r, terminal)
            time_step += 1
        
        rewards.append(total_reward.cpu().numpy())
        scores.append(score)

    return np.mean(rewards), np.mean(scores)


def play_game(options):
    """Play flappy bird with pretrained dqn model

       weight -- model file name containing weight of dqn
       best -- if the model is best or not
    """
    model = QNetwork()
    if options.ckpt_path is None:
        print ('you should give weight file name.')
        return
    print ('load previous model weight: {}'.format(options.ckpt_path))
    episode, epsilon = load_checkpoint(options.ckpt_path, model)

    # if options.cuda:
    #     model = model.cuda()

    algorithm = DQN(model, optim, epsilon, options)

    algorithm.set_eval()
    bird_game = game.GameState()
    bird_game.FPS = 30

    action = [1, 0]
    o, r, terminal = bird_game.frame_step(action)
    o = preprocess(o)

    rpm = ReplayMemory(1, options)
    rpm.append(o, action, r, terminal)

    start = time.time()
    fc = 0
    score = 0
    while True:
        prev_o, a, r, o, terminal = rpm.sample(1)

        # q = algorithm(o).cpu().detach().numpy()[0]

        score = max(score, bird_game.score)
        action = algorithm.get_optim_action(o)
        o, r, terminal = bird_game.frame_step(action)
        
        o = preprocess(o)

        # img = Image.fromarray((o*255).astype(np.uint8)).convert(mode='L')
        # img.save(f'{fc}-{r}-{q.argmax()}.png')
        # fc += 1
        if terminal or score > options.max_score*2:
            break

        rpm.append(o, action, r, terminal)

    ela = time.time() - start
    print(f'Final Score {score}, FPS{bird_game.FPS}, {ela//60}m{ela%60}s')
    
