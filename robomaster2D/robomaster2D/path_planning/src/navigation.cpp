#include "navigation.hpp"
#include <math.h>
#include <algorithm>
#include <iostream>

namespace critical_hit_planning
{
    bool Navigation::navigation_path(double robot_x, double robot_y, double robot_angle, 
                                     std::vector<path_point> final_path, 
                                     double &vx, double &vy, double &angular, double &path_angular)
    {
//        if(final_path.empty()){ return false;}
        calc_target_index(robot_x, robot_y, final_path);

        path_angular = atan2(final_path[current_target_id_].y_ - robot_y,
                                    final_path[current_target_id_].x_ - robot_x);

        double delta_yaw = path_angular - robot_angle;
        if(delta_yaw > 3.14159265358) delta_yaw -= 6.283185307;
        if(delta_yaw < -3.14159265358) delta_yaw += 6.283185307;
        if(delta_yaw < 0)
        {
            angular = trapezoidal_plan(angular, 0, omiga_max_, 10, -delta_yaw-0.01, -delta_yaw);
            angular *= -1;
        }
        angular = trapezoidal_plan(angular, 0, omiga_max_, 10, delta_yaw-0.01, delta_yaw);

        double tx = final_path[current_target_id_].x_;
        double ty = final_path[current_target_id_].y_;
        double tv = final_path[current_id_].v_;
        double alpha = atan2(ty - robot_y, tx - robot_x) - robot_angle;
        angular = 2.0 * vx * sin(alpha) / Lfc;
        if(angular > omiga_max_)
            angular = omiga_max_;
        double delta_v = a_max_ * 0.006;
        if(tv > vx)
        {
            vx += delta_v;
            if(vx > tv) vx = tv;
        }
        else
        {
            vx -= delta_v;
            if(vx < tv) vx = tv;
        }
        if(sqrt((tx-robot_x)*(tx-robot_x)+(ty-robot_y)*(ty-robot_y)) < 0.05)
        {
            return false;
        }
        return true;
    }

    void Navigation::calc_target_index(double robot_x, double robot_y, std::vector<path_point> final_path)
    {
        double min_dist = 100;
        double min_index = 0;
        for(int i = 0; i < final_path.size(); i++)
        {
            double dist = std::sqrt((final_path[i].x_-robot_x)*(final_path[i].x_-robot_x)+
                                    (final_path[i].y_-robot_y)*(final_path[i].y_-robot_y));
            if(dist < min_dist)
            {
                min_dist = dist;
                min_index = i;
            }
        }
        current_id_ = min_index;
        if(current_target_id_ == final_path.size() - 1) return;
        if(min_dist > Lfc)
        {
            current_target_id_ = current_id_;
            return;
        }
        double L = min_dist;
        while(Lfc > L && min_index < final_path.size() - 1)
        {
            double dx = final_path[min_index+1].x_-final_path[min_index].x_;
            double dy = final_path[min_index+1].y_-final_path[min_index].y_;
            L += std::sqrt(dx * dx + dy * dy);
            min_index += 1;
        }
        current_target_id_ = min_index;
    }

    double Navigation::trapezoidal_plan(double x_start, double x_end, double x_max, double acc, double d, double l)
    {
        if(l-d < 0)
            d = l - 0.01;
        if(d < 0)
            d = 0;
        double acc_d1 = (x_max*x_max-x_start*x_start)/(2*acc);
        double acc_d2 = (x_max*x_max-x_end*x_end)/(2*acc);
        if(acc_d1 + acc_d2 < l)
        {
            //   -------
            //  /       \
            // /         \
            // 这种情况下还能达到最大速度
            if(l-d < acc_d1)
            {
                return std::sqrt(x_start*x_start + 2*acc*(l-d));
            }
            else if(l-d < l-acc_d2)
            {
                return x_max;
            }
            else
            {
                return std::sqrt(x_end*x_end + 2*acc*d);
            }
        }
        else
        {
            double acc_d = fabs(x_start*x_start-x_end*x_end)/(2*acc);
            if(acc_d < l)
            {
                //    /\
                //   /  \
                //  /
                // 这种情况下不能达到最大速度，但是还有加速和减速阶段
                double max = std::sqrt((x_start*x_start+x_end*x_end+2*acc*l)/2);
                double d1 = (max*max-x_start*x_start)/(2*acc);
                if(l-d < d1)
                {
                    return std::sqrt(x_start*x_start+2*acc*(l-d));
                }
                else
                {
                    if(x_end*x_end+2*acc*d < 0) return 0;
                    return std::sqrt(x_end*x_end+2*acc*d);
                }
            }
            else
            {
                //    /
                //   /
                //  /
                // 这种情况下只有加速阶段或只有减速阶段
                if(x_start > x_end)
                {
                    return std::sqrt(x_end*x_end+2*acc*d);
                }
                else
                {
                    return std::sqrt(x_start*x_start+2*acc*(l-d));
                }
            }
        }
        
    }
}