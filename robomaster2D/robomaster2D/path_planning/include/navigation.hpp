#ifndef CRITICAL_HIT_NAVIGATION_H
#define CRITICAL_HIT_NAVIGATION_H

#include "smooth.hpp"

namespace critical_hit_planning
{
    class Navigation
    {
        public:
        Navigation()
        {
            v_max_ = 3.3;
            a_max_ = 5;
            omiga_max_ = 6.28;
            Lfc = 0.3;
            current_target_id_ = 0;
            current_id_ = 0;
        }
        bool navigation_path(double robot_x, double robot_y, double robot_angle, 
                             std::vector<path_point> final_path, 
                             double &vx, double &vy, double &angular, double &path_angular);
        void reset(double yaw)
        {
            current_target_id_ = 0;
            current_id_ = 0;
        }
        int current_target_id_;
        int current_id_;
        private:
        double v_max_;
        double a_max_;
        double omiga_max_;
        double Lfc;// 前视距离
        void calc_target_index(double robot_x, double robot_y, std::vector<path_point> final_path);
        double trapezoidal_plan(double x_start, double x_end, double x_max, double acc, double d, double l);
    };
}

#endif