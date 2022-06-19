import numpy as np
import time
import torch
import os


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
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # an early time
        self.save_interval = save_interval
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")
        self.epoch = 0

        self.record_controller_id = 0

    def evaluate_save(self, act, cri, steps=0, log_tuple=None, logger=None, enemy_act=None) -> bool:
        if log_tuple is None:
            log_tuple = [0, 0, 0]
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()
            rewards_steps_list = []
            infos_dict = {}
            self.env.env.display_characters("正在评估...")
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
                torch.save(act.state_dict(), os.path.dirname(os.path.realpath(__file__)) + '/' + act_save_path)
                act_save_path = f'{self.cwd}/critic_step:' + str(self.total_step) + '.pth'
                torch.save(cri.state_dict(), os.path.dirname(os.path.realpath(__file__)) + '/' + act_save_path)

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
            logger.log(train_infos, step=self.total_step)
            self.epoch += 1
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f" {'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                      f"  {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2} {self.total_step:8.2e} {self.target_return:8.2f} |"
                      f"{r_avg:8.2f} {r_std:8.2f} |"
                      f"  {self.used_time:>8}  ########")

            # plan to
            # if time.time() - self.print_time > self.show_gap:
            print(f" {self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                  f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                  f"{' '.join(f'{n:8.2f}' for n in log_tuple)} | "
                  f"{infos_dict['red_win_rate']:.2f}, {infos_dict['red_draw_rate']:.2f}")
        else:
            if_reach_goal = False

        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


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
