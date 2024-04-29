import sys
import argparse
import torch
from train import train_dqn, play_game

# 构建解析器
parser = argparse.ArgumentParser(description='DQN demo for flappy bird')

# 添加参数
parser.add_argument('--play', action='store_true', default=False,
                    help='If set False, train the model; otherwise, play game with pretrained model')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='If set true, with cuda enabled; otherwise, with CPU only')
parser.add_argument('--resume', action='store_true', default=False, help='resume training with pretrained model')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--gamma', type=float, help='discount rate', default=0.95)
parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--init_e', type=float, help='initial epsilon for epsilon-greedy exploration', default=1.0)
parser.add_argument('--final_e', type=float, help='final epsilon for epsilon-greedy exploration', default=0.01)
parser.add_argument('--exploration', type=int, help='number of exploration using epsilon-greedy policy', default=40000)
parser.add_argument('--max_score', type=int, help='if score exceeds max_score, stop training/evaluating', default=10000)
parser.add_argument('--max_episode', type=int, help='maximum episode of training', default=6000)
parser.add_argument('--rpm_size', type=int, help='size of replay memory', default=6000)
parser.add_argument('--learning_freq', type=int, help='learning frequency in run_episode', default=64)
parser.add_argument('--evaluate_freq', type=int, help='evaluate frequency in run_episode', default=500)
parser.add_argument('--update_target_steps', type=int, help='update frequency for target Q model', default=16)
parser.add_argument('--ckpt_path', type=str, help='weight file name for finetunig(Optional)',
                    default='ckpt/episode.ckpt')
parser.add_argument('--save_checkpoint_freq', type=int, help='episode interval to save checkpoint', default=2000)
parser.add_argument('--save_checkpoint_freq', type=int, help='episode interval to save checkpoint', default=2000)

if __name__ == '__main__':

    args = parser.parse_args()  # 解析参数
    print("the args is:", args)

    if args.cuda and not torch.cuda.is_available():
        print('CUDA is not available, maybe you should not set --cuda')
        sys.exit(1)

    if args.play and args.ckpt_path == '':
        print('When test, a pretrained weight model file should be given')
        sys.exit(1)

    if args.cuda:
        print('With GPU support!')

    if args.play:
        play_game(args)
    else:
        train_dqn(args)
