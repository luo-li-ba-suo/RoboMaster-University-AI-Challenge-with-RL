import sys

from elegantrl_dq.envs.env import *
from elegantrl_dq.agents.AgentPPO import *
from elegantrl_dq.train.run import *


def demo_discrete_action():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    env_name = 'Robomaster-v0'
    args.config.if_multi_processing = True
    args.config.new_processing_for_evaluation = False
    args.config.num_envs = 8
    args.config.if_wandb = False

    args.config.reward_scale = 2 ** -1
    args.config.net_dim = 128
    args.config.gamma = 0.98
    args.config.batch_size = 256
    args.config.repeat_times = 8
    args.config.repeat_times_policy = 8
    args.config.target_step = 4096
    args.config.learning_rate = 1e-4
    args.config.if_per_or_gae = True
    args.config.if_allow_break = False
    args.config.break_step = 10000000

    args.config.random_seed = 1

    args.config.eval_times1 = 5
    args.config.eval_times2 = 10

    args.config.if_use_cnn = True
    args.config.if_share_network = True

    args.config.if_print_time = True
    args.config.if_train = True
    args.config.self_play = False
    # args.config.cwd = '2022-07-24_16-44-21'
    # args.config.enemy_cwd = 'init_model'
    args.agent = MultiEnvDiscretePPO()
    if args.config.if_multi_processing and args.config.if_train:
        args.env = VecEnvironments(env_name, args.config.num_envs)
    else:
        args.env = VecEnvironments(env_name, 1)
    if not args.config.if_train:
        args.config.eval_gap = 0
        args.env_eval = PreprocessEnv(env_name)
    elif not args.config.new_processing_for_evaluation:
        args.config.eval_gap = 60 * 3
        args.env_eval = PreprocessEnv(env_name)
    args.agent.cri_target = False
    '''train and evaluate'''
    train_and_evaluate(args)


if __name__ == '__main__':
    demo_discrete_action()
