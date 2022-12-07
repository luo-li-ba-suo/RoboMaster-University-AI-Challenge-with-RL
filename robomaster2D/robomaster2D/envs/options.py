class Parameters:
    # 规则规定：
    cooling_freq = 10

    # 环境配置
    robot_r_num = 2
    robot_b_num = 2
    red_agents_name = 'Critical Hit'
    blue_agents_name = 'HITCSC'

    superQuiet = True
    show_poses = False
    do_route_plan = False
    buff_mode = False
    rotate_by_route_plan = False
    episode_time = 180
    episode_step = 128
    # episode_step如果不为零，当step数量达到这个值，将提前结束episode

    frame_num_one_step = 1
    frame_num_one_second = 20.0  # >= 20; 最大需求频率为20hz（50ms）
    overheating_protection = True

    start_pos = [[758, 398],
                 [758, 50],
                 [50, 50],
                 [50, 398]]
    random_start_pos = True
    random_start_far_pos = True
    random_start_far_dis = 350
    start_angle = [180, 90, 0, -90]
    start_bullet = [500, 500, 500, 500]

    unlimited_bullet = [False,
                        False,
                        False,
                        False]
    episodes = 100000
    render_per_frame = 600
    action_type = 'MultiDiscrete'
    obstacle_map_unit = 8
    obstacle_map_size = 100

    # 有关hp
    start_hp = [100, 100, 100, 100]
    no_dying = [False, False]
    collision_reduce_hp = False

    # 有关物理效果
    impact_effect = True
    collision_bounce = False

    # 有关交互界面
    do_plot = [True, True, False, False]
    energySaving_mode = False

    highFrameRate_mode = True
    show_robot_points = False
    show_center_barrier_vertices = False
    show_goals_position = False
    show_goal_line = False
    show_state_data = True
    show_robot_data = True
    show_figure = True

    # priority init
    random_start_prob = 1
    priority_init_capacity = 1000

    # 有关测试：
    render = False
    training_mode = False
    single_input = False
    time_delay_frame = 0

    # 有關訓練
    red_agents_path = ['src.agents.rl_trainer']
    blue_agents_path = ['src.agents.handcrafted_enemy', 'src.agents.retreat_enemy']
    eval_blue_agents_path = ['src.agents.handcrafted_enemy', 'src.agents.retreat_enemy']

    # 有关地图信息：
    enable_blocks = True

    # 有关Astar：
    Astar_map_x_size = 81
    Astar_map_y_size = 45
    Astar_obstacle_expand = 2

    # 有关雷达
    use_lidar = True
    # 雷达数要求为偶数
    lidar_num = 30

    # 有关局部障碍物地图
    use_obstacle_map = False

    def get_dict(self):
        dict_ = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('__') and not callable(value):
                dict_[name] = value
        return dict_
