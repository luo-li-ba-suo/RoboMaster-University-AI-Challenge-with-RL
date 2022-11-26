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
    args.config.learning_rate = 2e-4
    args.config.if_per_or_gae = True
    args.config.if_allow_break = False
    args.config.break_step = 100000000

    args.config.random_seed = 1

    args.config.eval_times = args.config.num_envs * 2

    args.config.if_use_cnn = True
    args.config.if_use_rnn = True
    args.config.sequence_length = 2 ** 3
    args.config.if_share_network = False

    args.config.if_print_time = True
    args.config.if_train = True

    args.config.self_play = False
    args.config.enemy_act_update_interval = 2 ** 12
    args.config.model_pool_capacity_historySP = 1000
    args.config.delta_historySP = 1.0
    args.config.enemy_stochastic_policy = True

    args.config.use_extra_state_for_critic = True
    args.config.use_action_prediction = True

    args.config.frame_stack_num = 4
    args.config.history_action_stack_num = 3
    # args.config.cwd = '2022-07-24_16-44-21'
    # args.config.enemy_cwd = 'init_model'
    args.agent = MultiEnvDiscretePPO()
    args.env = VecEnvironments(env_name, args.config.num_envs,
                               pseudo_step=1 if args.config.use_action_prediction else 0)
    args.config.eval_gap = 60 * 3
    args.env_eval = VecEnvironments(env_name, args.config.num_envs, seed_offset=100)
    args.agent.cri_target = False
    '''train and evaluate'''
    train_and_evaluate(args)


if __name__ == '__main__':
    demo_discrete_action()
