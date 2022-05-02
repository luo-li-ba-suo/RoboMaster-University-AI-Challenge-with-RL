class Parameters:
    robot_r_num = 1
    robot_b_num = 1
    red_agents_path = 'src.agents.rl_agent'
    blue_agents_path = 'src.agents.random_agent'
    red_agents_name = 'Critical Hit'
    blue_agents_name = 'HITCSC'
    render = False
    superQuiet = True
    show_poses = False
    do_route_plan = False
    buff_mode = False
    rotate_by_route_plan = False
    episode_time = 180
    episode_step = 128
    # episode_step如果不为零，当step数量达到这个值，将提前结束episode
    cooling_freq = 10
    frame_num_one_time = 2
    frame_num_one_second = 200
    overheating_protection = True

    # 有关地图信息：
    enable_blocks = False

    start_pos = [[758, 398],
                 [758, 50],
                 [50, 50],
                 [50, 398]]
    random_start_pos = True
    random_start_far_pos = True
    start_angle = [180, 90, 0, -90]
    start_bullet = [500, 0, 500, 0]

    unlimited_bullet = [False,
                        False,
                        False,
                        False]
    episodes = 100000
    render_per_frame = 600
    action_type = 'MultiDiscrete'
    local_map_unit = 10
    local_map_size = 28

    # 有关hp
    start_hp = [100, 100, 100, 100]
    no_dying = [False, False]
    collision_reduce_hp = False

    # 有关物理效果
    impact_effect = True
    collision_bounce = True

    # 有關信息處理
    lidar_num = 8

    # 有关交互界面
    do_plot = [True, False, False, False]
    节能模式 = False
    训练模式 = True
    高帧率模式 = True
    show_robot_points = False
    show_center_barrier_vertices = False
    show_goals_position = False
    show_goal_line = False
    show_state_data = True
    show_robot_data = True
    show_figure = True

    # 有关测试：
    single_input = False
    time_delay_frame = 0

    # 有關訓練
    controller_ids = [0]