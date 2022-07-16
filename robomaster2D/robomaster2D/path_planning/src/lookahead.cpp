#include "lookahead.hpp"
#include <math.h>
#include <algorithm>
#include <iostream>

namespace critical_hit_planning
{
    LookAhead::LookAhead()
    {
        v_max_ = 3.3;
        a_max_ = 5;
        omiga_max_ = 6.28;
    }

    std::vector<path_point> LookAhead::update_velocity(double current_v, std::vector<smooth_path> path)
    {
        std::vector<path_point> final_path;
        // 起始速度
        path[0].start_speed_ = current_v;
        // 终止速度
        path[path.size()-1].end_speed_ = 0;
        for(int i = 1; i < path.size()-1; i+=2)
        {
            // 路径都是一段直线-一段圆弧这样的排列顺序
            if(path[i].is_arc_ == true)
            {
                // 按最大旋转速度计算圆弧初始线速度
                double v = path[i].r_ * omiga_max_;
                
                // 上一个直线段对线速度的限制
                if(v*v-path[i-1].start_speed_*path[i-1].start_speed_ > 2*a_max_*path[i-1].len_)
                {
                    v = std::sqrt(path[i-1].start_speed_*path[i-1].start_speed_+2*a_max_*path[i-1].len_);
                }
                if(path[i-1].start_speed_*path[i-1].start_speed_-v*v > 2*a_max_*path[i-1].len_)
                {
                    v = std::sqrt(path[i-1].start_speed_*path[i-1].start_speed_-2*a_max_*path[i-1].len_);
                }
                // 初始线速度不能超过最大限制
                if(v > v_max_)
                {
                    v = v_max_;
                }
                if(v < 0)
                {
                    v = 0;
                }
                path[i-1].end_speed_ = v;
                path[i+1].start_speed_ = v;
                path[i].start_speed_ = v;
                path[i].end_speed_ = v;
            }
        }
        // 检查最后一段是否满足动力学要求
        if(path[path.size()-1].start_speed_*path[path.size()-1].start_speed_ > 2*a_max_*path[path.size()-1].len_)
        {
            // 如果不满足，则前推更新速度
            for(int i = path.size() - 2; i > 0; i-=2)
            {
                double v;
                if(path[i+1].start_speed_*path[i+1].start_speed_-path[i+1].end_speed_*path[i+1].end_speed_ > 2*a_max_*path[i+1].len_)
                {
                    v = std::sqrt(path[i+1].end_speed_*path[i+1].end_speed_+2*a_max_*path[i+1].len_);
                    if(v > v_max_)
                    {
                        v = v_max_;
                    }
                    path[i+1].start_speed_ = v;
                    path[i].start_speed_ = v;
                    path[i].end_speed_ = v;
                    path[i-1].end_speed_ = v;
                }
                else
                {
                    break;
                }
            }
        }
        double point_step = 0.1;
        for(int i = 0; i < path.size(); i++)
        {
            double total_step = 0;
            double tmp_x, tmp_y, tmp_v;
            while(total_step < path[i].len_)
            {
                if(point_step > path[i].len_)
                {
                    total_step += point_step;
                    continue;
                }
                if(path[i].is_arc_ == false)
                {
                    total_step += point_step;
                    if(total_step > path[i].len_) total_step = path[i].len_;
                    tmp_x = (path[i].end_x_ - path[i].start_x_) * total_step / path[i].len_ + path[i].start_x_;
                    tmp_y = (path[i].end_y_ - path[i].start_y_) * total_step / path[i].len_ + path[i].start_y_;
                    tmp_v = trapezoidal_plan(path[i].start_speed_, path[i].end_speed_, v_max_, a_max_, 
                                             path[i].len_-total_step, path[i].len_);
                    path_point p(tmp_x, tmp_y, tmp_v);
                    final_path.push_back(p);
                }
                else
                {
                    total_step += point_step;
                    if(total_step > path[i].len_) total_step = path[i].len_;
                    double delta_step = total_step / path[i].r_;
                    double delta_angle = path[i].end_angle_ - path[i].start_angle_;
                    if(delta_angle > 3.14159265358) delta_angle -= 6.283185307;
                    if(delta_angle < -3.14159265358) delta_angle += 6.283185307;
                    double step_angle = (path[i].end_angle_ - path[i].start_angle_) * delta_step / fabs(delta_angle) + path[i].start_angle_;
                    tmp_v = path[i].start_speed_;
                    tmp_x = path[i].center_x_ + path[i].r_ * cos(step_angle);
                    tmp_y = path[i].center_y_ + path[i].r_ * sin(step_angle);
                    path_point p(tmp_x, tmp_y, tmp_v);
                    final_path.push_back(p);
                }
            }
        }
        return final_path;
    }

    double LookAhead::trapezoidal_plan(double x_start, double x_end, double x_max, double acc, double d, double l)
    {
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
                if(x_start*x_start + 2*acc*(l-d) < 0) return 0;
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
                    if(x_start*x_start+2*acc*(l-d) < 0) return 0;
                    return std::sqrt(x_start*x_start+2*acc*(l-d));
                }
                else
                {
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
                    if(x_start*x_start+2*acc*(l-d) < 0) return 0;
                    return std::sqrt(x_start*x_start+2*acc*(l-d));
                }
            }
        }
    }
}