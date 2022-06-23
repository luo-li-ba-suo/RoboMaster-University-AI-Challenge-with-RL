import sys

from elegantrl_dq.envs.env import *
from elegantrl_dq.agents.AgentPPO import *
from elegantrl_dq.train.run import *


def demo_discrete_action():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    env_name = 'Robomaster-v0'
    args.if_multi_processing = True
    args.new_processing_for_evaluation = False
    args.num_envs = 8
    args.if_wandb = True
    if args.if_multi_processing:
        args.agent = MultiEnvDiscretePPO()
        args.env = VecEnvironments(env_name, args.num_envs)
    else:
        args.agent = AgentDiscretePPO()
        args.env = PreprocessEnv(env_name)  # 表示训练所有trainer中的第一个，其他trainer会一起共享模型
    if not args.new_processing_for_evaluation:
        args.env_eval = PreprocessEnv(env_name)
    args.agent.cri_target = False
    args.reward_scale = 2 ** -1
    args.net_dim = 128
    args.gamma = 0.98
    args.batch_size = 1024
    args.repeat_times = 8
    args.repeat_times_policy = 8
    args.target_step = 4096
    args.learning_rate = 1e-4
    args.if_per_or_gae = True
    args.if_allow_break = False
    args.break_step = 20000000
    args.gpu_id = sys.argv[-1][-4]
    args.random_seed = 1

    args.eval_gap = 60
    args.eval_times1 = 5
    args.eval_times2 = 3

    args.if_print_time = True
    args.if_train = True
    # args.cwd = '2022-05-14_21-13-49-perfect'
    # args.enemy_cwd = '2022-05-10_17-33-26'
    '''train and evaluate'''
    train_and_evaluate(args)


if __name__ == '__main__':
    demo_discrete_action()
