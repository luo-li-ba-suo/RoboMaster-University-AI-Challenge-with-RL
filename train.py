import sys

from elegantrl_dq.envs.env import *
from elegantrl_dq.agents.AgentPPO import *
from elegantrl_dq.train.run import *

if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU cannot be used')


def demo_discrete_action():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    if_train_robomaster = 1
    if if_train_robomaster:
        args.env = PreprocessEnv(env=gym.make('Robomaster-v0'))
        args.agent.trainer_id = args.env.env.trainer_ids[0]  # 表示训练所有trainer中的第一个，其他trainer会一起共享模型
        args.agent.cri_target = False
        args.reward_scale = 2 ** -1
        args.net_dim = 128
        args.gamma = 0.98
        args.batch_size = 256
        args.repeat_times = 16
        args.repeat_times_policy = 5
        args.target_step = 4096
        args.learning_rate = 1e-4
        args.if_per_or_gae = True
        args.if_allow_break = False
        args.break_step = 20000000
        args.gpu_id = sys.argv[-1][-4]
        args.random_seed = 1

        args.eval_gap = 2 ** 6
        args.eval_times1 = 20
        args.eval_times2 = 30

        args.if_train = True
        # args.cwd = '2022-05-14_21-13-49-perfect'
        # args.enemy_cwd = '2022-05-10_17-33-26'
    '''train and evaluate'''
    train_and_evaluate(args)


if __name__ == '__main__':
    demo_discrete_action()
