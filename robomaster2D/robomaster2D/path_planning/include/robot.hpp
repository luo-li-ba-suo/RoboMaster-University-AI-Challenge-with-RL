#ifndef SIMULATOR_ROBOT_H
#define SIMULATOR_ROBOT_H

#include "navigation.h"
#include "global_planning.h"

using namespace hitcrt;
namespace hitsz_icra_simulator{
    class Robot
    {
        public:
        Robot();
        void set_goal(int pos_x, int pos_y);// 设置机器人的目标坐标，启动路径规划
        void path_plan();// 被move函数调用，进行路径规划
        double pos_[2];
        double chassis_angle_ = 0;// 底盘角度，逆时针为正，范围是-180-180，角度值
        double speed_x_ = 0;
        double speed_y_ = 0;
        double angular_ = 0;// 角速度
        NavInfo nav_info;// 导航类
        GlobalPlanner g_planner;// 全局路径规划类
        vector<vector<float>> path_position;
        vector<NavInfo::ST_3V> path, last_path;
        bool check_obstacle(vector<MapSearchNode> obstacle);
        void add_obstacle(double x, double y, double width, double height);
        void path_init();
        bool segment(double* p1, double* p2, double *p3, double *p4);
        double cross(double* p1, double* p2, double *p3);
        void ChangePath();
        bool isNav();
    };
}
#endif