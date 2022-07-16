#include "smooth.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace critical_hit_planning
{
    std::vector<smooth_path> SmoothCorner::smooth_sharp_corner(cv::Mat costmap, std::vector<struct Point> astar_path)
    {
        // 简化路径，共线的点只保留两边的俩端点
        astar_path = simplify_path(astar_path);
        // debug:弗洛伊德路径平滑
        for(int i = astar_path.size()-1; i > 0; i--){
            for(int j = 0; j < i-1; j++){
                if(CheckObstacle(costmap, astar_path[i], astar_path[j])){//判断2点之间是否有障碍物
                    for(int k = i-1; k > j; k--){
                        astar_path.erase(astar_path.begin()+k);//删除多余的拐点
                    }
                    i = j;
                    break;
                }
            }
        }

        smooth_path_.clear();
        // 只有两个点，直接存储一条直线
        if(astar_path.size() == 2)
        {
            smooth_path_.push_back(smooth_path(astar_path[0].x, astar_path[0].y, astar_path[1].x, astar_path[1].y));
        }
        else
        {
            // 多于两个点，需要圆弧插补
            struct Point last_end, end_p;
            last_end.x = astar_path[0].x;
            last_end.y = astar_path[0].y;
            // 开始插补
            for(int i = 1; i < astar_path.size()-1; i++)
            {
                struct Point vector_1, vector_2, vector_3;
                double vector_len;
                double delta = delta_max_;
                // 上一段的向量
                vector_1.x = last_end.x - astar_path[i].x;
                vector_1.y = last_end.y - astar_path[i].y;
                vector_len = std::sqrt(vector_1.x*vector_1.x+vector_1.y*vector_1.y);
                double l1 = vector_len;
                vector_1.x /= vector_len;vector_1.y /= vector_len;
                // 下一段的向量
                vector_2.x = astar_path[i+1].x - astar_path[i].x;
                vector_2.y = astar_path[i+1].y - astar_path[i].y;
                vector_len = std::sqrt(vector_2.x*vector_2.x+vector_2.y*vector_2.y);
                double l2 = vector_len;
                vector_2.x /= vector_len;vector_2.y /= vector_len;
                // 转折处的夹角
                double beta = acos(vector_1.x*vector_2.x+vector_1.y*vector_2.y);
                struct Point start_p, center;
                
                // 计算路径方向误差
                double d = delta*cos(beta/2)/(1-sin(beta/2));
                if(d > epsilon_) d = epsilon_;
                if(d > l1/2) d = l1/2;
                if(d > l2/2) d = l2/2;
                // 计算垂直路径方向的误差
                delta = d * (1-sin(beta/2))/cos(beta/2);
                // 圆弧半径
                double r = d * tan(beta/2);
                start_p.x = astar_path[i].x + d * vector_1.x;
                start_p.y = astar_path[i].y + d * vector_1.y;
                end_p.x = astar_path[i].x + d * vector_2.x;
                end_p.y = astar_path[i].y + d * vector_2.y;
                // 转折点指向圆弧圆心的向量
                vector_3.x = vector_1.x + vector_2.x;
                vector_3.y = vector_1.y + vector_2.y;
                vector_len = std::sqrt(vector_3.x*vector_3.x+vector_3.y*vector_3.y);
                vector_3.x /= vector_len;vector_3.y /= vector_len;
                center.x = astar_path[i].x + (r + delta) * vector_3.x;
                center.y = astar_path[i].y + (r + delta) * vector_3.y;
                double start_angle = atan2(start_p.y - center.y, start_p.x - center.x);
                double end_angle = atan2(end_p.y - center.y, end_p.x - center.x);
                if(start_angle - end_angle > 3.141592653) start_angle -= 6.283185307;
                if(start_angle - end_angle < -3.14159265) start_angle += 6.283185307;
                // 按照顺序，先保存直线段，再保存圆弧段
                // 如果是最后一个转折点，则把最后一段直线也保存了
                smooth_path_.push_back(smooth_path(last_end.x, last_end.y, start_p.x, start_p.y));
                smooth_path_.push_back(smooth_path(center.x, center.y, r, start_angle, end_angle));
                if(i == astar_path.size()-2)
                {
                    smooth_path_.push_back(smooth_path(end_p.x, end_p.y, astar_path[i+1].x, astar_path[i+1].y));
                }
                last_end = end_p;
            }
        }
        return smooth_path_;
    }

    std::vector<struct Point> SmoothCorner::simplify_path(std::vector<struct Point> astar_path)
    {
        if(astar_path.size() <= 2) return astar_path;
        std::vector<struct Point> simple_path;
        double last_k = atan2(astar_path[1].y-astar_path[0].y, astar_path[1].x-astar_path[0].x);
        simple_path.push_back(astar_path[0]);
        for(int i = 2; i < astar_path.size(); i++)
        {
            double k = atan2(astar_path[i].y-astar_path[i-1].y, astar_path[i].x-astar_path[i-1].x);
            if(fabs(k-last_k) < 0.0001)
                continue;
            simple_path.push_back(astar_path[i-1]);
            last_k = k;
        }
        simple_path.push_back(astar_path[astar_path.size()-1]);
        return simple_path;
    }

    bool SmoothCorner::CheckObstacle(cv::Mat costmap, Point p1, Point p2)
    {
        double vector_x = (p1.x - p2.x);
        double vector_y = (p1.y - p2.y);
        int point_num = 10;
        bool symble = true;
        for(int i=1;i<=point_num;i++){
            double x = p2.x + vector_x*i/(point_num+1);
            double y = p2.y + vector_y*i/(point_num+1);
            int pixel_x = x*20;
            int pixel_y = costmap.rows - y*20;
            if(255 - costmap.at<unsigned char>(pixel_y, pixel_x) > 20){//出现障碍则返回false
                symble = false;
                break;
            }
        }
        
        return symble;
    }
}