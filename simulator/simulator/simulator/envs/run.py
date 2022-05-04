# import argparse
import kernel_game
import sys
from options import Parameters
sys.path.append('./simulator/')


# def loadParameter():
#     description = """
#     USAGE:      python run.py <options>
#     """
#     parser = argparse.ArgumentParser(description)
#
#     parser.add_argument('-r', '--robot_r_num', type=int, help='Red team agent num', default=2)
#     parser.add_argument('-b', '--robot_b_num', type=int, help='Blue team agent num', default=2)
#     parser.add_argument('--red_agents_path', type=str, help='red_agents_path',
#                         default='src.agents.human_agent')
#     # parser.add_argument('--red_agents_path', type=str, help='red_agents_path',
#     #                     default='./src/agents/rl_trainer.py')
#     parser.add_argument('--blue_agents_path', type=str, help='blue_agents_path',
#                         default='src.agents.human_agent')
#     parser.add_argument('--red_agents_name', type=str, help='red_agents_name', default='Critical Hit')
#     parser.add_argument('--blue_agents_name', type=str, help='blue_agents_name', default='HITCSC')
#     parser.add_argument('--render', type=bool, help='do render', default=True)
#     parser.add_argument('--superQuiet', type=bool, help='superQuiet', default=True)
#     parser.add_argument('--show_poses', type=bool, help='show_poses', default=False)
#     parser.add_argument('--impact_effect', type=bool, help='impact_effect', default=True)
#     parser.add_argument('--do_route_plan', type=bool, help='do_route_plan', default=False)
#     parser.add_argument('--buff_mode', type=bool, help='buff_mode', default=True)
#     parser.add_argument('--rotate_by_route_plan', type=bool, help='rotate_by_route_plan', default=False)
#     parser.add_argument('--episode_time', type=int, help='', default=180)
#     parser.add_argument('--cooling_freq', type=int, help='', default=10)
#     parser.add_argument('--frame_num_one_time', type=int, help='how many frames in one time', default=10)
#     parser.add_argument('--unlimited_bullet_list', type=list, help='', default=[True, True, True, True])
#     parser.add_argument('--start_pos', type=list, help='', default=[[758, 398, 180, 50],
#                                                                     [758, 50, 90, 0],
#                                                                     [50, 50, 0, 50],
#                                                                     [50, 398, -90, 0]])
#     parser.add_argument('--unlimited_bullet', type=list, help='', default=[False,
#                                                                            False,
#                                                                            False,
#                                                                            False])
#
#     ##  强化学习
#     parser.add_argument('--episodes', type=int, help='episodes', default=100000)
#     parser.add_argument('--action_type', type=str, help='Discrete action or Partly continuous action(acceleration)',
#                         default='Discrete')
#     ##  局部地图
#     parser.add_argument('--local_map_unit', type=int, help='local_map_unit', default=10)
#     parser.add_argument('--local_map_size', type=int, help='local_map_size', default=28)
#
#     return parser.parse_args()




if __name__ == '__main__':
    args = Parameters()
    args.red_agents_path = 'src.agents.human_agent'
    args.blue_agents_path = 'src.agents.human_agent'
    args.render_per_frame = 20
    args.render = True
    simulator = kernel_game.Simulator(args)
    simulator.state.pause = True
    state = simulator.reset()
    for e in range(args.episodes):
        done = simulator.step([])
        if done:
            state = simulator.reset()
        if simulator.state.do_render is False:
            print('Simulator closed')
            break
