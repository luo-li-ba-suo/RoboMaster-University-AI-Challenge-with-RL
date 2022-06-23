import sys
import gym
import simulator
from elegantrl.envs.Gym import PreprocessEnv
from elegantrl.train.run import *
from elegantrl.agents.agent import *
from elegantrl.train.config import Arguments


'''demo'''


def demo_discrete_action_off_policy():
    env_name = ['Robomaster-v0',
                'LunarLander-v2', ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU
    env = PreprocessEnv(env=gym.make(env_name))
    args = Arguments(env=env, agent=AgentDQN)
    args.reward_scale = 2 ** -1
    args.net_dim = 64
    args.gamma = 0.97
    args.batch_size = 128
    # args.target_step = args.max_step
    # args.learning_rate = 1e-5
    # args.if_use_per = True
    # args.if_allow_break = False
    # args.break_step = args.target_step * 50000
    # args.gpu_id = sys.argv[-1][-4]
    args.random_seed = 1

    args.eval_gap = 2 ** 6
    # args.eval_times =

    args.learner_gpus = gpu_id
    # args.random_seed += gpu_id



    if_check = 1
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = 0
    ENV_ID = 0

    # demo_continuous_action_off_policy()
    # demo_continuous_action_on_policy()
    demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
