#include <iostream>
#include "planning.hpp"

// 与python交互的一些函数
namespace critical_hit_planning
{
    double pos_x, pos_y, pos_angle;
    double vx = 0, vy = 0, angular = 0, path_angular=0;
    Planning plan;
    extern "C"
    {
        void init(bool render = true, bool block_map = false)
        {
            plan.show_map = render;
            plan.load_map(block_map);
            std::cout << std::endl;
            std::cout << "|****************HITSZ Critical HIT**************|" << std::endl;
            std::cout << "|*******RMUA 2021 SIMULATOR Path Planning********|" << std::endl;
            std::cout << std::endl;
        }

        int set_goal(int x, int y)
        {
//                std::cout<<"set_goal: x:"<<x<<" y:"<<y<<"\n";
            plan.set_goal((double)x/100, (double)(448-y)/100);
        }

        double get_robot_vx()
        {
            return vx * cos(-path_angular);
        }

        double get_robot_vy()
        {
            return vx * sin(-path_angular);
        }

        double get_robot_angular()
        {
            return -angular/ 3.1415926 * 180;
        }

        void update_pos(double x, double y, double angle)
        {
            pos_x = x/100;
            pos_y = (448 - y)/100;
            pos_angle = -angle * 3.1415926 / 180;
        }

        bool path_plan()
        {
            return plan.path_plan(vx, vy, angular, path_angular, pos_x, pos_y, pos_angle);
//            std::cout<<"angle:"<<pos_angle<<"\n";
        }

        int isNav()
        {
            if(plan.Is_Nav())
                return 1;
            else
            {
                vx = 0;
                vy = 0;
                angular = 0;
                return 0;
            }
        }

        void add_obstacle(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4)
        {
            plan.add_block(x1/5, y1/5, x2/5, y2/5, x3/5, y3/5, x4/5, y4/5);
        }
        void clean_obstacle()
        {
            plan.clean_block();
        }

        int** get_block_map()
        {
            static int block_map[45][81];
            plan.get_block_map(block_map);
            return (int **)block_map;
        }
    }
}