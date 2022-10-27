import importlib
import numpy as np


class AgentsAllocator:
    def __init__(self, args):
        if not args.red_agents_path or not args.blue_agents_path:
            raise '---------No agents specified----------'
        self.file_list = [args.red_agents_path, args.blue_agents_path, args.eval_blue_agents_path]
        self.red_team = []
        self.blue_team = []
        self.eval_blue_team = []
        for i, team in enumerate(self.file_list):
            for agent_file_path in team:
                agent_file_path = 'robomaster2D.envs.' + agent_file_path
                print(f"path is {agent_file_path}")
                try:
                    my_module = importlib.import_module(agent_file_path)
                    agent_temp = my_module.My_Agent(i, args)
                    if i == 0:
                        self.red_team.append(agent_temp)
                    elif i == 1:
                        self.blue_team.append(agent_temp)
                    else:
                        self.eval_blue_team.append(agent_temp)

                    if not args.superQuiet:
                        print('Team {} agent {} loaded'.format(i, agent_file_path))
                except (NameError, ImportError, IOError):
                    raise 'Error: The agent "' + agent_file_path + '" could not be loaded! '
                except BaseException as e:
                    raise e

    def get_agents(self):
        agents = []
        if len(self.red_team) == 1:
            agents.append(self.red_team[0])
        else:
            agents.append(self.red_team[np.random.randint(0, len(self.red_team))])
        if len(self.blue_team) == 1:
            agents.append(self.blue_team[0])
        else:
            agents.append(self.blue_team[np.random.randint(0, len(self.blue_team))])
        return agents

    def get_eval_agents(self):
        agents = []
        if len(self.red_team) == 1:
            agents.append(self.red_team[0])
        else:
            agents.append(self.red_team[np.random.randint(0, len(self.red_team))])
        if len(self.eval_blue_team) == 1:
            agents.append(self.eval_blue_team[0])
        else:
            agents.append(self.eval_blue_team[np.random.randint(0, len(self.eval_blue_team))])
        return agents
