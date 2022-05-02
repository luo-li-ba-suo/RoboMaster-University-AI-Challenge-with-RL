#include <iostream>
#include "robot.hpp"
// 与python交互的一些函数
namespace hitsz_icra_simulator
{
    Robot robot;
    extern "C"
    {
        void init()
        {
            std::cout << std::endl;
            std::cout << "|********HITSZ Critical HIT********|" << std::endl;
            std::cout << "|*******ICRA 2021 SIMULATOR********|" << std::endl;
            std::cout << std::endl;
        }

        void set_goal(int x, int y)
        {
            robot.set_goal(x, y);
        }

        double get_robot_vx()
        {
            return robot.speed_x_;
        }

        double get_robot_vy()
        {
            return robot.speed_y_;
        }

        double get_robot_angular()
        {
            return robot.angular_;
        }

        int isNav()
        {
            if(robot.isNav())
                return 1;
            else
            {
                robot.speed_x_ = 0;
                robot.speed_y_ = 0;
                robot.angular_ = 0;
                return 0;
            }
        }

        void update_robot_info(double x, double y, double angle)
        {
            robot.pos_[0] = x;
            robot.pos_[1] = y;
            robot.chassis_angle_ = angle;
        }

        void path_plan()
        {
            robot.path_plan();
        }

        void path_init()
        {
            robot.path_init();
        }

        void add_obstacle(double x, double y, double width, double height)
        {
            robot.add_obstacle(x, y, width, height);
        }
    }
}