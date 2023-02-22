# Robomaster AI Challenge仿真环境

视频演示：
https://m.bilibili.com/video/BV1Xe411N7rC

## 下一步计划关键部分：
计划参考捉迷藏工程
TODO：在见不到对方之前瞄准
dead state to zeros
### 对障碍物进行建模
- 构建含障碍物的局部地图
- critic和actor网络共享CNN主干
### 构建行为树
- handcrafted enemy可以基于Astar路径规划模块对我方进行分别跟踪打击


## 强化学习部分
- 奖励：
  - TODO：增加与障碍物碰撞的惩罚
- TODO: update obs space(such as velocity)
- actor网络最后一层权重用小std正交初始化
- 改relu为tanh后效果反变差
- entropy loss暂时去掉

## 环境部分
- 新增障碍物类型。场地中有两种不同高度的障碍物，较矮的障碍物只能阻挡机器人前进不能阻挡子弹，标记为1,其他的标记为0
- 进入robomaster2D, 使用指令"pip install -e ."就可以安装环境;运行./robomaster2D/robomaster2D/envs文件夹下的RMUA_Env_for_RL.py文件即可手动控制机器人来测试环境
- 防止打死队友：瞄准敌人之后才允许射击（action mask）
- 解决bug：如果子弹同时穿过障碍物与机器人，就会判定子弹撞上机器人，没有先后
  - 解决办法：在有障碍物的情况下，将子弹轨迹分成多段判断
- 尚存bug：某个或多个打击源同时致死时，可能导致伤害溢出，这样计算出来的分别的伤害值是不准确的
## 评估部分
- 已解决：当前胜率为环境内置的前100场胜率，应改为实时胜率
- 评估进程可能是内存溢出的元凶;长期训练应不采用评估进程

## 与神经网络控制的敌人对抗训练部分
- 敌人模型有两种模式：
  - 与自己的模型同步（优先级：1）
    - 可调节同步间隔
  - 读取固定策略模型（优先级：2）
- 评估进程敌人的策略模式：
  - 固定策略：只读取一次模型，敌人策略网络无需共享内存
  - 变化策略
- 评估进程是否创建敌人模型，取决于是否给定敌人策略的路径或是否self play
## 多进程运行环境部分
- 多进程方面如果某个管道堆积太多消息可能会导致cpu内存爆炸
- 会开一个监视进程，负责记录主进程发送了几次消息给评估进程，作为共享变量交给评估进程利用，评估进程就可以做相应次数的接收来获取最新消息
- 多进程环境无法开启pygame渲染，因为fork不支持pygame.font
## 可视化部分
- 环境的参数中可配置是否渲染环境界面。
  - 若配置为render=True，则在调用环境时可通过env.render等函数来开启渲染等功能;
  - 若配置为render=False，则调用env.render等函数无法开启渲染等功能
- 为了使得渲染画面更流畅，并能够自由指定渲染帧率，RMUA_Env_for_RL中的env.render函数只是用来开启渲染。开启渲染后，各个模块中的render_inited属性将被激活，从而进行对应的UI显示操作

- RMUA_Env_for_RL中的display_character函数可以指定一串字符串进行显示，多种模式开发中



## 历史更改记录（弃）
###运行说明
命令行运行：python kernal_game.py 可键盘操控模拟环境

命令行运行：python ICRA2021_env.py 开始dqn训练

###V1.7更新记录：
1.网络中添加了一层隐藏层，隐藏层为128维
2.每次训练时时间（游戏内部时间）为30s
3.修正了在禁止移动惩罚区不能旋转炮管和自瞄的bug
4.修正了先被禁止射击后被禁止移动时，等到禁止移动结束才能射击的bug

###V1.8更新记录：
1.增加了在gpu上训练神经网络的功能
2.增加了保存模型和读取模型的功能
3.归一化网络输入的状态值
4.改为任意一个机器人死亡即停止训练而非都死亡才停止训练

###V1.9更新记录：
1.采用自动瞄准模式训练，减少了云台转动的行为空间，并修复了瞬间自瞄、有些情况自瞄失败的bug
2.修改机器人轮子、装甲撞击墙或其他机器人的状态为实时值而非累积值，以便奖励制定
3.添加实时奖励状态，当机器人获取地图上buff加成时触发
4.添加实时被打击状态，被击中头、侧边、尾部装甲时分别为1,2,3
5.修改奖励,去除奖励中血量剩余、弹丸剩余的部分，只留下了实时状态
6.修改状态空间，添加了敌方的坐标和角度，但没有添加buff分布，奖励中的buff加成或惩罚也已去除
7.修改奖励,增加发弹状态并加入负奖励
8.旧版本中游戏环境跑10次才返回一次状态值，导致奖励计算时10次漏掉9次状态，大大影响训练效果;目前已修复bug，但跑的很慢

###V1.10更新记录：
1.模拟环境修改：模拟环境中每一个step跑10次epoch，每10次epoch状态的累积值算作一次step的实时状态值并用作reward计算（如轮子撞击次数、被击中次数、发弹次数），并在结束step时清空累积值

###V0.0更新记录：
1.从小车导航开始训练，将buff去除，并取消小车与障碍物碰撞效果
2.在靠近障碍物200mm时会有负奖励
3.机器人不会死亡
4.q网络输入精简为x y angle，动作空间精简为x y angle

###V0.1更新记录：
1.增加reward曲线
2.去掉地图边缘碰撞效果，防止因为碰撞使得小车超出地图时获得奖励高于碰到障碍物的时候
3.限制机器人不能离开地图边缘10单位


###V2.0 局部地图作为输入 更新记录：
1.网格化地图并生成状态输入，采取机器人周围局部状态地图输入到卷积层网络，其他状态信息输入全连接层
2.支持d*lite导航
3.激活撞击效果
4.修复了重复归一化的bug
5.能够显示机器人周围视野
6.在机器人局部地图维度发生异常时输出状态信息并拒绝存入buffer
7.将模型测试代码独立出来

###V2.1 更新记录：
1.在v2版本每轮随机生成目标点

###V2.2 hierachy 更新记录：
1.支持自动识别gpu进行训练，可以在cpu设备上运行

2.决策层分为两层：实时决策层和子目标点选择层

实时决策层：输出是否设置子目标点的动作

子目标点选择层：输出局部地图中子目标点

3.底层控制层用d*lite实现

4.随机生成目标位置并显示，**该版本可训练到达随机目标点的任务**

5.游戏环境修改：

由于导航模块直接输出的是速度值，所以order类添加了速度模式，在order转化为act时判断是否为速度模式，速度是则直接把赋值给act（乘以常量）

###V2.3 d* lite 更新记录
1.游戏环境更新自瞄函数：

增加是否能击中对方的判断和自动射击

注意游戏环境中：
**time单位为秒，1s=200epoch，指令获取（orders_to_acts）为0.05s一次，枪管冷却为0.1s一次,move_robot为0.005s一次, shoot_mutiple为0.005s一次,非shoot_mutiple为0.01s一次**

2.设置两个敌人自动追击我方，一旦能击中我方则停止追击, **该版本可训练目标点选择的任务**

3.设置我方用分层强化学习训练，动作空间为全局地图目标点：以大致５０分米为网格边长栅格化，得到全局８４个点，作为我方训练的动作空间

4.决策层分为两层：实时决策层和子目标点选择层

实时决策层：输出是否设置子目标点的动作

子目标点选择层：输出全局地图中目标点

5.状态空间包括变量如下：

x坐标，y坐标，b1 x坐标，y坐标，b2 x坐标，y坐标，b1 b2 vx,vy(相对地图), hp loss, b1 b2 hp loss, b1 b2 lidar(是否能雷达检测到敌方)

实施决策层的输入量还包括子目标点坐标

###V2.4 four frames 更新记录
1.该版本训练强化学习控制底层的任务，其中输入为四帧状态，包括局部地图;只有一层决策层，输出底层动作

2.输出的横纵坐标动作为相对地图的动作，不再输出机器人转向和射击，机身转向、云台转向和射击均由游戏引擎自动控制

3.除局部地图外输入状态量除上一版本外加入了碰撞参数

4.修改网络，去除池化层，增加对位置的敏感度

5.对机器人类增加了敌人导致的扣血量这一属性


### V2.4.1 four frames 更新记录
1.训练1v1撤退任务


### V2.7 Rainbow-random-goal 更新记录
1.采用Rainbow算法训练到达随机目标点任务：  
<font color=red>***参数:***  
***状态空间：*** 坐标，与目标点横纵坐标距离，目标点坐标，墙面撞击，28x28 局部地图  
***动作空间：*** 3x3邻域  
***batch_size:*** 64  
***target_update:*** 200</font>  
2.修改代码结构，kernel部分放在单独文件夹  
3.将env环境代码独立出来，并重写初始化代码，不同的任务由不同的初始化函数执行  
4.可以实时显示plot结果  
5.可固定随机数  
<font color=yellow>6.env中不需要在done时自动reset，env的reset交由agent来调取</font>  
7.废除[0,0]原地不动的动作  
8.set reward as follows:  
hit -> -r  
dis -> -r  
act -> -2(wrong moving dir), 0(one moving dir is correct but the other is wrong moving dir), 2(correct moving dir)
9.注意游戏环境中order获取时，forward_mode为True保证上下左右的动作不受机器人朝向影响  

###<font color=yellow>V2.7训练效果</font>  
八十万frames后：  
1.训练出来了到达目标点的效果 --> 说明奖励、方法合理  
2.但没有学到躲避障碍物 --> 卷积层最多才32层，可能太少了；对障碍物的奖励设计可能还不够合理，应该靠近障碍物即给负奖励或者越靠近障碍给越多负奖励  
3.由于游戏环境的影响，机器人容易卡在障碍物尖角处 --> 需完善游戏环境  
4.训练后期缺乏探索，机器人容易卡在难以到达目标点的地方  

### V2.8 Rainbow-d*lite 更新记录
1.采用Rainbow算法和d*lite训练对抗任务，继承v2.3  
2.注意游戏环境中order获取时，v_mode为True保证获取的为速度而非上下左右的方向
3.我方和敌人刷在随机点，reset时需要把mode_home设为True，且赋值start_pos  
4.用路径规划时，注意order第三个参数要赋值angular, auto_rotate为True使得机身旋转不受路径规划控制  
5.自瞄模块转动机身时往需要更大转动角度的方向转，需要修正

### V2.9 Asynchronous-Rainbow-d*lite 更新记录  
1.采用异步学习和异步采集的rainbow分别生成一个版本


###<font color=yellow>V4.0 Asynchronous-Rainbow-d*lite 更新记录  </font>  
###计划
1.首先彻底翻新模拟环境,解耦各个模块：  
 a.扩大模拟环境界面，使可显示机器人信息、血条等多方面信息  
 b.检查和校验机器人发弹、移动的数据，使之更贴近实际；比如发弹过快、发弹命中率设计  
 c.设计迟钝的敌人行为树  
 d.完善环境的动作层、行为树层、多智能体功能的整理  
2.设计阶段性训练任务：  
 a.停在基地不动的敌人（发弹和不发弹）  
 b.随机行走的敌人（发弹迟钝和不发弹）  
 c.跟踪我方的敌人（发弹迟钝和灵敏）  
3.仿真动作空间重新设计：  
 a.自瞄改换成选择攻击目标后瞄准射击，对于无法攻击到的敌方用动作mask处理  
 b.实现离散和连续的动作空间灵活切换，进而为dqn系列和ac系列切换提供便利  
4.设计目标空间，应用her算法  

###改动
1.游戏环境  
 a.规则中描述50ms检测一次装甲板，由于现实中子弹给装甲板的力是持续的，50ms内只要有子弹撞击而且无论多少子弹撞击都能且只能检测到一次撞击，然而在仿真中的撞击是在一帧中完成的，于是50ms检测一次装甲板转化为50ms最多只能检测到一次最大的撞击  

###<font color=yellow>V4.04 PPO  </font>  
1.奖励增加：    
 离buff距离  
 捡到对方buff惩罚  
2.动作增加：  
 瞄准射击  
<font color=yellow>3.agent添加动作掩膜</font>  
4.增加帧率切换按钮  
<font color=yellow>5.loss图</font>  
6.显示：  
 <font color=yellow>击中敌人</font> 
7.状态信息：  
 死亡机器人值为零with agent ID
8.训练
对照PPO论文总结改进
 epoch改小
 minibatch 改大
 clip 0.1~0.2
9.注意一个问题：
  如果因热量掉血快速死亡拿到的总奖励高于存活且靠近敌人拿到的总奖励，智能体会快速死亡
  如果学到了因射击而掉血，可能见到敌人都不射击了（需要对射击赋予正奖励）

###<font color=yellow>V4.114 PPO  </font>  
1.增加功能并重构step和frame之间的关系：
将用户控制模式中在停止控制的时候暂停游戏的功能放在step函数中
