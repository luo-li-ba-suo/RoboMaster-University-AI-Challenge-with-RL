import sys

from elegantrl_dq.envs.env import *
from elegantrl_dq.agents.AgentPPO import *
from elegantrl_dq.train.run import *


def demo_discrete_action():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    env_name = 'Robomaster-v0'
    args.config.if_multi_processing = True
    args.config.new_processing_for_evaluation = False
    args.config.num_envs = 10
    args.config.if_wandb = False

    args.config.reward_scale = 2 ** -1
    args.config.net_dim = 256
    args.config.gamma = 0.998
    args.config.batch_size = 2 ** 12
    args.config.repeat_times = 4
    args.config.repeat_times_policy = 4
    args.config.target_step = 2 ** 16
    args.config.learning_rate = 1e-4
    args.config.if_per_or_gae = True
    args.config.if_allow_break = False
    args.config.break_step = 10000000

    args.config.random_seed = 1

    args.config.eval_times1 = 8
    args.config.eval_times2 = 16

    args.config.if_use_cnn = True
    args.config.if_share_network = False

    args.config.if_print_time = True
    args.config.if_train = True
    args.config.self_play = False
    # args.config.cwd = '2022-07-24_16-44-21'
    # args.config.enemy_cwd = 'init_model'
    args.agent = MultiEnvDiscretePPO()
    if args.config.if_multi_processing and args.config.if_train:
        args.env = VecEnvironments(env_name, args.config.num_envs,
                                   pseudo_step=1 if args.config.use_action_prediction else 0)
    else:
        args.env = VecEnvironments(env_name, 1,
                                   pseudo_step=1 if args.config.use_action_prediction else 0)
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
