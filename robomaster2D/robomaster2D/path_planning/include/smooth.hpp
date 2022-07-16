#ifndef CRITICAL_HIT_SMOOTH_H
#define CRITICAL_HIT_SMOOTH_H

#include "build_map.hpp"
#include <math.h>
#include <algorithm>

namespace critical_hit_planning
{
    class path_point
    {
        public:
        path_point(double x, double y, double v)
        {
            x_ = x;
            y_ = y;
            v_ = v;
        }
        double x_;
        double y_;
        double v_;
    };
    class smooth_path
    {
        public:
        smooth_path(double start_x,double start_y,double end_x,double end_y)
        {
            start_x_ = start_x;
            start_y_ = start_y;
            end_x_ = end_x;
            end_y_ = end_y;
            len_ = std::sqrt((start_x_-end_x_)*(start_x_-end_x_)+(start_y_-end_y_)*(start_y_-end_y_));
            is_arc_ = false;
        }
        smooth_path(double center_x, double center_y, double r, double start_angle, double end_angle)
        {
            center_x_ = center_x;
            center_y_ = center_y;
            r_ = r;
            start_angle_ = start_angle;
            end_angle_ = end_angle;
            double delta_angle = end_angle - start_angle;
            if(delta_angle > 3.14159265358) delta_angle -= 6.283185307;
            if(delta_angle < -3.14159265358) delta_angle += 6.283185307;
            len_ = r * fabs(delta_angle);
            is_arc_ = true;
        }
        double start_x_;
        double start_y_;
        double end_x_;
        double end_y_;
        double len_;
        double center_x_;
        double center_y_;
        bool is_arc_;
        double start_angle_;
        double end_angle_;
        double r_;
        double start_speed_;
        double end_speed_;
    };
    class SmoothCorner
    {
        public:
        void set_delta_max(double delta_max)
        {
            delta_max_ = delta_max;
        }
        std::vector<smooth_path> smooth_sharp_corner(cv::Mat costmap, std::vector<struct Point> astar_path);
        private:
        double delta_max_;
        double epsilon_ = 0.4;
        std::vector<smooth_path> smooth_path_;
        std::vector<struct Point> simplify_path(std::vector<struct Point> astar_path);
        bool CheckObstacle(cv::Mat costmap, Point p1, Point p2);
    };
}

#endif