import time
import torch
import os
from elegantrl_dq.envs.env import *
from elegantrl_dq.agents.net import *

class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device, save_interval):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # an early time
        self.save_interval = save_interval
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")
        self.epoch = 0

        self.record_controller_id = 0

    def evaluate_save(self, act, cri, steps=0, log_tuple=None, logger=None, enemy_act=None):
        if log_tuple is None:
            log_tuple = [0, 0, 0]
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()
            rewards_steps_list = []
            infos_dict = {}
            self.env.display_characters("正在评估...")
            for _ in range(self.eval_times1):
                reward, step, reward_dict, info_dict = get_episode_return_and_step(self.env, act, self.device,
                                                                                   enemy_act)
                rewards_steps_list.append((reward, step))
                for key in reward_dict:
                    if 'reward_' + key in infos_dict:
                        infos_dict['reward_' + key].append(reward_dict[key])
                    else:
                        infos_dict['reward_' + key] = [reward_dict[key]]
                for key in info_dict:
                    if key[-1] != '_':
                        if 'red_' + key in infos_dict:
                            infos_dict['red_' + key].append(info_dict[key])
                        else:
                            infos_dict['red_' + key] = [info_dict[key]]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                for _ in range(self.eval_times2 - self.eval_times1):
                    reward, step, reward_dict, info_dict = get_episode_return_and_step(self.env, act, self.device,
                                                                                       enemy_act)
                    rewards_steps_list.append((reward, step))
                    for key in reward_dict:
                        infos_dict['reward_' + key].append(reward_dict[key])
                    for key in info_dict:
                        if key[-1] != '_':
                            infos_dict['red_' + key].append(info_dict[key])
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            for key in infos_dict:
                infos_dict[key] = np.mean(infos_dict[key])
            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)
                # logger.save_state({'env': env}, None)
                '''save policy network in *.pth'''
                act_save_path = f'{self.cwd}/actor_best.pth'
                # print('current dir:'+os.path.dirname(os.path.realpath(__file__)))
                # print('act_save_path:'+act_save_path)
                torch.save(act.state_dict(), act_save_path)
                act_save_path = f'{self.cwd}/actor_step:' + str(self.total_step) + '_best.pth'
                torch.save(act.state_dict(), act_save_path)
                if logger:
                    logger.save(act_save_path)

                print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |")  # save policy and print
            elif not self.epoch % self.save_interval:
                '''save policy network in *.pth'''
                act_save_path = f'{self.cwd}/actor_step:' + str(self.total_step) + '.pth'
                torch.save(act.state_dict(), act_save_path)
                act_save_path = f'{self.cwd}/critic_step:' + str(self.total_step) + '.pth'
                torch.save(cri.state_dict(), act_save_path)

            '''save record in logger'''
            train_infos = {'Epoch': self.epoch,
                           str(self.agent_id) + '_MaxR': self.r_max,
                           str(self.agent_id) + '_avgR': log_tuple[3],
                           str(self.agent_id) + '_avgR_eval': r_avg,
                           str(self.agent_id) + '_stdR_eval': r_std,
                           str(self.agent_id) + '_avgS_eval': s_avg,
                           str(self.agent_id) + '_stdS_eval': s_std,
                           str(self.agent_id) + '_objC': log_tuple[0],
                           str(self.agent_id) + '_objA': log_tuple[1],
                           str(self.agent_id) + '_logprob': log_tuple[2]}
            train_infos.update(infos_dict)
            if logger:
                logger.log(train_infos, step=self.total_step)
            self.epoch += 1
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f" {'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                      f"  {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2} {self.total_step:8.2e} |"
                      f"{r_avg:8.2f} {r_std:8.2f} |"
                      f"  {self.used_time:>8}  ########")

            # plan to
            # if time.time() - self.print_time > self.show_gap:
            print(f" {self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                  f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                  f"{' '.join(f'{n:8.2f}' for n in log_tuple)} | "
                  f"{infos_dict['red_win_rate']:.2f}, {infos_dict['red_draw_rate']:.2f}")

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


class AsyncEvaluator:
    def __init__(self, models, cwd, agent_id, eval_times1, eval_times2, eval_gap, env_name, device, save_interval,
                 logger):
        self.logger = logger
        env = PreprocessEnv(env_name)
        env.render()
        self.device = device
        self.evaluator = Evaluator(cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device, save_interval)
        _mp = mp.get_context("spawn")
        self.evaluator_conn, self.env_conn = mp.Pipe()
        process = _mp.Process(target=self.run, args=(models,))
        process.start()
        self.env_conn.close()

    def run(self, models):
        act = models['act']
        cri = models['cri']
        enemy_act = models['enemy_act']
        net_dim = models['net_dim']
        state_dim = models['state_dim']
        action_dim = models['action_dim']
        if_build_enemy_act = models['if_build_enemy_act']
        local_act = MultiAgentActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device)
        local_enemy_act = MultiAgentActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device) \
            if if_build_enemy_act else None
        local_cri = CriticAdv(net_dim, state_dim).to(self.device)
        local_act.eval()
        local_cri.eval()
        if if_build_enemy_act:
            local_enemy_act.eval()
        self.evaluator_conn.close()
        self.env_conn.send('start')
        while True:
            request, package = self.env_conn.recv()
            if request == "update":
                steps, logging_tuple = package
                local_act.load_state_dict(act.state_dict())
                local_cri.load_state_dict(cri.state_dict())
                if if_build_enemy_act:
                    local_enemy_act.load_state_dict(enemy_act.state_dict())

                self.evaluator.evaluate_save(local_act, local_cri, enemy_act=local_enemy_act, logger=self.logger,
                                             steps=steps, log_tuple=logging_tuple)
                self.env_conn.send('finish')
            else:
                raise NotImplementedError

    def update(self, steps, logging_tuple):
        message = self.evaluator_conn.recv()
        if message == "finish" or "start":
            self.evaluator_conn.send(("update", (steps, logging_tuple)))
            return True
        return False

def get_episode_return_and_step(env, act, device, enemy_act=None) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    info_dict = None
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete
    if_multi_discrete = env.if_multi_discrete
    state = env.reset()
    trainer_ids_in_the_start = env.env.trainer_ids

    for episode_step in range(max_step):
        a_tensor = [None for _ in range(env.env.simulator.state.robot_num)]
        for i in range(env.env.simulator.state.robot_num):
            if i in env.env.trainer_ids:
                s_tensor = torch.as_tensor((state[i],), device=device)
                a_tensor[i] = act(s_tensor)
            elif i in env.env.nn_enemy_ids:
                s_tensor = torch.as_tensor((state[i],), device=device)
                a_tensor[i] = enemy_act(s_tensor)
        # if if_discrete:
        #     a_tensor = a_tensor.argmax(dim=1)
        #     action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        actions = [None for _ in range(env.env.simulator.state.robot_num)]
        if if_multi_discrete:
            for i, a_tensor_ in enumerate(a_tensor):
                if a_tensor_ is not None:
                    n = 0
                    action = []
                    for action_dim_ in env.action_dim:
                        action.append(a_tensor_[:, n:n + action_dim_].argmax(dim=1).detach().cpu().numpy()[0])
                        n += action_dim_
                    actions[i] = action
        state, rewards, done, info_dict = env.step(actions)
        episode_return += np.mean(rewards)
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    rewards_dict = [env.env.rewards_episode[i] for i in trainer_ids_in_the_start]
    mean_reward_dict = {}
    for key in rewards_dict[0]:
        mean_reward_dict[key] = np.mean([r[key] for r in rewards_dict])
    return episode_return, episode_step, mean_reward_dict, info_dict
