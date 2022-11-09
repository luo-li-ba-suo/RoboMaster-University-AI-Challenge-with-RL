import signal

from elegantrl_dq.train.replay_buffer import *
from elegantrl_dq.train.evaluator import *
from elegantrl_dq.utils.process_log import process_info


class Configs:
    def __init__(self, gpu_id=None, if_on_policy=False):
        self.cwd = time.strftime("%Y-%m-%d_%H-%M-%S",
                                 time.localtime())  # current work directory. cwd is None means set it automatically
        self.enemy_cwd = None
        self.if_train = True
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.if_print_time = False
        '''Arguments for multi-processing'''
        self.if_multi_processing = True
        self.num_envs = 2
        self.new_processing_for_evaluation = False
        '''Arguments for env'''
        self.env_name = None
        self.env_config = None

        '''Arguments for training (off-policy)'''
        self.if_use_cnn = True
        self.if_use_rnn = False
        self.if_share_network = True
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.train_actor_step = 0
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 60 * 10  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.break_step = 2 ** 2  # break training after 'total_step > break_step'
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.save_interval = 25  # save the model per how many times of evaluation

        '''Arguments for algorithm'''
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.lambda_gae_adv = 0.98

        '''Arguments for self play '''
        self.self_play = True
        # self play mode:
        # naive self play: 0
        # history self play : 1
        # PSRO: 2
        self.self_play_mode = 1
        self.model_pool_capacity_historySP = 100
        self.delta_historySP = 1.0
        self.fix_evaluation_enemy_policy = True
        self.enemy_act_update_interval = 0

        '''Arguments for extra state for critic '''
        self.use_extra_state_for_critic = True
        self.use_action_prediction = True

        '''Arguments for wandb'''
        self.if_wandb = True
        self.wandb_user = 'dujinqi'
        self.wandb_notes = 'lidar'
        self.wandb_name = 'actionPrediction' + str(self.random_seed)
        self.wandb_group = None  # 是否障碍物地图
        self.wandb_job_type = None  # 是否神经网络控制的敌人


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.config = Configs(gpu_id=gpu_id, if_on_policy=if_on_policy)

    def init_before_training(self, process_id=0):
        self.config.env_name = self.env.env_name
        self.config.env_config = self.env.args.get_dict()

        if self.config.env_config['enable_blocks']:
            self.config.wandb_group = 'ObstacleMap'
        else:
            self.config.wandb_group = 'NoObstacleMap'
        self.config.wandb_job_type = 'conv_' if self.config.if_use_cnn else ''
        self.config.wandb_job_type += 'sharedAC' if self.config.if_share_network else 'separatedAC'

        # ppo
        if hasattr(self.agent, 'ratio_clip'):
            self.agent.ratio_clip = self.config.ratio_clip
        if hasattr(self.agent, 'lambda_entropy'):
            self.agent.lambda_entropy = self.config.lambda_entropy
        if hasattr(self.agent, 'lambda_gae_adv'):
            self.agent.lambda_gae_adv = self.config.lambda_gae_adv

        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set None value automatically'''
        if self.config.gpu_id is None:  # set gpu_id as '0' in default
            self.config.gpu_id = '0'

        if self.config.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.config.cwd = f'./{self.env.env_name}_{agent_name}'

        if process_id == 0:
            print(f'| GPU id: {self.config.gpu_id}, cwd: {self.config.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.config.if_remove is None:
                self.config.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.config.cwd)) == 'y')
            if self.config.if_remove:
                shutil.rmtree(self.config.cwd, ignore_errors=True)
                print("| Remove history")
            # os.makedirs(self.cwd, exist_ok=True)
        least_target_step = self.config.num_envs * self.config.env_config['episode_step']
        agent_num = 0
        if 'src.agents.rl_trainer' in self.config.env_config['red_agents_path']:
            agent_num += self.config.env_config['robot_r_num']
        if 'src.agents.rl_trainer' in self.config.env_config['blue_agents_path']:
            agent_num += self.config.env_config['robot_b_num']
        assert self.config.target_step > agent_num*least_target_step, "too small target_step, some bug will happen"
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.set_num_threads(self.config.num_threads)
        torch.set_default_dtype(torch.float32)

        gpu_id = self.config.gpu_id[process_id] if isinstance(self.config.gpu_id, list) else self.config.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        '''self play'''
        if self.config.self_play and self.config.self_play_mode == 1:
            assert self.config.env_config['blue_agents_path'] == ['src.agents.nn_enemy'], 'opponent should not be rl trainer'

        '''action prediction'''
        if not self.config.use_extra_state_for_critic:
            assert not self.config.use_action_prediction
        else:
            assert self.config.use_action_prediction
        if self.config.use_action_prediction:
            assert not self.config.if_share_network, "Shared AC Net do not support action prediction temporarily"

def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = os.path.join('./results/models/', args.config.cwd)
    cwd = os.path.normpath(cwd)
    if args.config.enemy_cwd:
        enemy_cwd = os.path.join('./results/models/', args.config.enemy_cwd)
        enemy_cwd = os.path.normpath(enemy_cwd)
        if_build_enemy_act = True
    else:
        enemy_cwd = None
        if_build_enemy_act = False
    env = args.env
    env_eval = args.env_eval
    if_wandb = args.config.if_wandb
    if_print_time = args.config.if_print_time
    agent = args.agent
    gpu_id = args.config.gpu_id
    if_multi_processing = args.config.if_multi_processing
    new_processing_for_evaluation = args.config.new_processing_for_evaluation
    configs = args.config

    if_train = args.config.if_train
    if if_train:
        os.makedirs(cwd, exist_ok=True)
        if if_wandb and not new_processing_for_evaluation:
            import wandb
            '''保存数据'''
            log_dir = Path("./results/wandb_logs") / args.env.env_name / 'NoObstacle' / 'ppo'
            os.makedirs(log_dir, exist_ok=True)
            wandb_run = wandb.init(config=configs,
                                   project=configs.env_name,
                                   entity=configs.wandb_user,
                                   notes=configs.wandb_notes,
                                   name=configs.wandb_name,
                                   group=configs.wandb_group,
                                   dir=log_dir,
                                   job_type=configs.wandb_job_type,
                                   reinit=True)
            wandb_run.config.update(configs.env_config)
        else:
            wandb_run = None
    else:
        wandb_run = None
    '''training arguments'''
    net_dim = args.config.net_dim
    if_use_cnn = args.config.if_use_cnn
    if_use_conv1D = args.env.args.use_lidar
    if_use_rnn = args.config.if_use_rnn
    if_share_network = args.config.if_share_network
    # max_memo = args.config.max_memo
    break_step = args.config.break_step
    batch_size = args.config.batch_size
    target_step = args.config.target_step
    repeat_times = args.config.repeat_times
    repeat_times_policy = args.config.repeat_times_policy
    learning_rate = args.config.learning_rate
    if_per_or_gae = args.config.if_per_or_gae
    enemy_act_update_interval = args.config.enemy_act_update_interval

    # 有关上帝视角critic：
    action_prediction_dim = args.env.action_dim.copy()
    action_prediction_dim += 1
    extra_state_kwargs = {'use_extra_state_for_critic': args.config.use_extra_state_for_critic,
                          'use_action_prediction': args.config.use_action_prediction,
                          'agent_num': args.env.args.robot_r_num + args.env.args.robot_b_num,
                          'action_prediction_dim': action_prediction_dim}

    self_play = args.config.self_play
    self_play_mode = args.config.self_play_mode
    delta_historySP = args.config.delta_historySP
    model_pool_capacity_historySP = args.config.model_pool_capacity_historySP
    fix_evaluation_enemy_policy = args.config.fix_evaluation_enemy_policy

    gamma = args.config.gamma
    reward_scale = args.config.reward_scale
    soft_update_tau = args.config.soft_update_tau
    # 开始训练actor的step
    train_actor_step = args.config.train_actor_step
    '''evaluating arguments'''
    show_gap = args.config.eval_gap
    eval_times1 = args.config.eval_times1
    eval_times2 = args.config.eval_times2
    save_interval = args.config.save_interval

    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    # env.render()
    # env_eval.render()
    if not if_multi_processing or not if_train:
        env_eval.render()
    max_step = env.max_step
    state_dim = env.state_dim
    observation_matrix_shape = env.observation_matrix_shape
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    if_multi_discrete = env.if_multi_discrete
    '''selfPlay args'''
    self_play_args = {'self_play': self_play,
                      'if_build_enemy_act': if_build_enemy_act,
                      'enemy_policy_share_memory': not fix_evaluation_enemy_policy and new_processing_for_evaluation,
                      'enemy_act_update_interval': enemy_act_update_interval,
                      'self_play_mode': self_play_mode,
                      'delta_historySP': delta_historySP,
                      'model_pool_capacity_historySP': model_pool_capacity_historySP}

    '''init: Agent, ReplayBuffer, Evaluator'''
    total_trainers_envs = agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, max_step=max_step,
               if_use_conv1D=if_use_conv1D,
               env=env, if_use_cnn=if_use_cnn, if_use_rnn=if_use_rnn,
               if_share_network=if_share_network, if_new_proc_eval=new_processing_for_evaluation,
               observation_matrix_shape=observation_matrix_shape, **self_play_args, **extra_state_kwargs)

    buffer_len = target_step + max_step
    async_evaluator = evaluator = None

    if if_train and new_processing_for_evaluation:
        async_evaluator = AsyncEvaluator(models=agent.models, cwd=cwd, agent_id=gpu_id, device=agent.device,
                                         env_name=env.env_name,
                                         eval_times1=eval_times1, eval_times2=eval_times2,
                                         save_interval=save_interval, configs=configs,
                                         fix_enemy_policy=fix_evaluation_enemy_policy,
                                         if_use_cnn=if_use_cnn, if_share_network=True)
    else:
        evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                              eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap,
                              save_interval=save_interval, if_train=if_train, gamma=gamma,
                              fix_enemy_policy=fix_evaluation_enemy_policy,
                              if_use_cnn=if_use_cnn, if_share_network=True)  # build Evaluator
    if if_multi_processing and if_train:
        buffer = PlugInReplayBuffer(env=env, max_len=buffer_len, state_dim=state_dim,
                                    total_trainers_envs=total_trainers_envs,
                                    action_dim=action_dim, observation_matrix_shape=observation_matrix_shape,
                                    if_discrete=if_discrete, if_multi_discrete=if_multi_discrete,
                                    if_use_cnn=if_use_cnn, if_use_rnn=if_use_rnn,
                                    **extra_state_kwargs)
    else:
        buffer = ReplayBuffer(max_len=buffer_len, state_dim=state_dim, action_dim=action_dim,
                              if_discrete=if_discrete, if_multi_discrete=if_multi_discrete)
    '''prepare for training'''
    agent.save_load_model(cwd=cwd, if_save=False)  # 读取上一次训练模型
    if if_build_enemy_act:
        agent.load_enemy_model(cwd=enemy_cwd)
    total_step = 0
    '''testing'''
    if not if_train:
        while True:
            evaluator.evaluate_save(agent.act, agent.cri, enemy_act=agent.enemy_act)
    '''start training'''
    if_train_actor = False if train_actor_step > 0 else True
    start_training = time.time()
    while if_train:
        if if_print_time:
            start_explore = time.time()
        with torch.no_grad():
            logging_list, step = agent.explore_env(env, target_step, reward_scale, gamma, buffer)
        if if_print_time:
            print(f'| ExploreUsedTime: {time.time() - start_explore:.0f}s')
            # if time.time() - start_explore > 50:
            #     break
            start_update_net = time.time()
        # steps = buffer.extend_buffer_from_list(trajectory_list)
        total_step += step
        if total_step > train_actor_step and not if_train_actor:
            if_train_actor = True
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, repeat_times_policy, soft_update_tau,
                                         if_train_actor)
        if if_print_time:
            print(f'| UpdateNetUsedTime: {time.time() - start_update_net:.0f}s')
            start_evaluate = time.time()
        logging_tuple = list(logging_tuple)
        logging_tuple += logging_list
        with torch.no_grad():
            if not new_processing_for_evaluation:
                evaluator.evaluate_save(agent.act, agent.cri, agent.act_optimizer, agent.cri_optimizer, total_step,
                                        logging_tuple, wandb_run,
                                        enemy_act=agent.enemy_act)
            else:
                async_evaluator.update(total_step, logging_tuple)
            if_train = not (total_step >= break_step or os.path.exists(f'{cwd}/stop'))
        if if_print_time:
            print(f'| EvaluateUsedTime: {time.time() - start_evaluate:.0f}s')
            print(process_info())
    print(f'| **** Training Finished **** | UsedTime: {time.time() - start_training:.0f}s | SavedDir: {cwd}')
    env.stop()
    if wandb_run:
        wandb_run.finish()
    # os.kill(int(process_info()['pid']), signal.SIGKILL)
