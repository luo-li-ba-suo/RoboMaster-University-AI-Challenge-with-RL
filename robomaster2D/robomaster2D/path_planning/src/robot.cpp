#include "robot.hpp"
#include <math.h>
#include <iostream>

double PI = 3.14159265357;

namespace hitsz_icra_simulator
{
    Robot::Robot()
    {
        g_planner.PlannerInit();
        nav_info.PIDInit();
        nav_info.target_angle = PI;
        nav_info.nav_state = NavInfo::NAV_OFF;
    }

    void Robot::set_goal(int pos_x, int pos_y)
    {
        g_planner.PlannerDelete();
        g_planner.PlannerInit();
        vector<float> StartPos = {(float)pos_[0]/100.f, 4.48f-(float)pos_[1]/100.f};
	    vector<float> EndPos = {pos_x/100.f, 4.48f-pos_y/100.f};
        path_position = g_planner.PlannerStart(StartPos, EndPos);
        NavInfo::ST_3V point;
        vector<NavInfo::ST_3V> path;
        for(uint i = 0; i < path_position[0].size(); i++)
        {
            point.x = path_position[0][i] * 1000;
            point.y = path_position[1][i] * 1000;
            path.push_back(point);
        }
        nav_info.path_pos = path;
        nav_info.nav_state = NavInfo::NAV_LINE;
        nav_info.NavigationInit();
    }

    void Robot::path_plan()
    {
        if(nav_info.nav_state == NavInfo::NAV_OFF)
            return;
        nav_info.position.x = pos_[0] * 10;
        nav_info.position.y = 4480-pos_[1] * 10;
        nav_info.position.angle = -chassis_angle_ * PI / 180;
        nav_info.RobotNavigation();
        for(int i = 0; i < g_planner.obstacles.size(); i++)
        {
            if(check_obstacle(g_planner.obstacles[i]))
            {
                ChangePath();
                return;
            }
        }
        speed_x_ = nav_info.vel.x / 10;
        speed_y_ = -nav_info.vel.y / 10;
        angular_ = -nav_info.vel.angle * 180 / PI;
    }

    bool Robot::isNav()
    {
        if(nav_info.nav_state == NavInfo::NAV_OFF)
            return false;
        else return true;
    }

    void Robot::path_init()
    {
        g_planner.refreshObstacle();
    }

    void Robot::add_obstacle(double x, double y, double width, double height)
    {
        g_planner.addObstacle(x,y,width,height);
        for(int k = 0; k < g_planner.nodes.size(); k++)
        {
            double dist = sqrt((g_planner.nodes[k].x-x)*(g_planner.nodes[k].x-x)+
                                (g_planner.nodes[k].y-y)*(g_planner.nodes[k].y-y));
            if(dist < 0.9)
                g_planner.PlannerAddPunish(k);
        }
    }

    double Robot::cross(double* p1, double* p2, double *p3)
	{
		double x1 = p2[0] - p1[0];
		double y1 = p2[1] - p1[1];
		double x2 = p3[0] - p1[0];
		double y2 = p3[1] - p1[1];
		return x1 * y2 - x2 * y1;
	}

    bool Robot::segment(double* p1, double* p2, double *p3, double *p4)
	{
		if(max(p1[0], p2[0])>min(p3[0], p4[0]) && max(p3[0], p4[0])>min(p1[0], p2[0])
		&& max(p1[1], p2[1])>min(p3[1], p4[1]) && max(p3[1], p4[1])>min(p1[1], p2[1]))
		{
			if(cross(p1,p2,p3)*cross(p1,p2,p4)<0 && cross(p3,p4,p1)*cross(p3,p4,p2)<0)
				return true;
			else
				return false;
		}
		else return false;
	}

    bool Robot::check_obstacle(vector<MapSearchNode> obstacle)
    {
        float obstacle_x = (obstacle[0].x + obstacle[3].x)/2;
        float obstacle_y = (obstacle[0].y + obstacle[3].y)/2;
        float width_x = (obstacle[3].x - obstacle[0].x)/2;
        float width_y = (obstacle[3].y - obstacle[0].y)/2;
        // 判断机器人是否阻挡当前路径
        std::vector<std::vector<std::vector<double>>> path_tmp;
        bool line_flag = 1;
        for(uint i = 0; i < nav_info.linecircle_path.size() - 1; i++)
        {
            if(line_flag)
            {
                double sx = nav_info.linecircle_path[i].x / 1000;
                double sy = nav_info.linecircle_path[i].y / 1000;
                double gx = nav_info.linecircle_path[i + 1].x / 1000;
                double gy = nav_info.linecircle_path[i + 1].y / 1000;
                std::vector<std::vector<double>> tmp;
                tmp.push_back({sx, sy});
                tmp.push_back({gx, gy});
                path_tmp.push_back(tmp);
            }
            else
            {
                double sx = nav_info.linecircle_path[i].x / 1000;
                double sy = nav_info.linecircle_path[i].y / 1000;
                double gx = nav_info.linecircle_path[i + 2].x / 1000;
                double gy = nav_info.linecircle_path[i + 2].y / 1000;
                std::vector<std::vector<double>> tmp;
                tmp.push_back({sx, sy});
                tmp.push_back({gx, gy});
                path_tmp.push_back(tmp);
                i++;
            }
            line_flag = !line_flag;
        }
        double p1[2] = {obstacle_x-width_x, obstacle_y-width_y};
        double p2[2] = {obstacle_x+width_x, obstacle_y+width_y};
        double p3[2] = {obstacle_x+width_x, obstacle_y-width_y};
        double p4[2] = {obstacle_x-width_x, obstacle_y+width_y};
        for(int i = 0; i < path_tmp.size(); i++)
        {
            // 先判断是否有端点在矩形内
            double l1[2] = {path_tmp[i][0][0], path_tmp[i][0][1]};
            double l2[2] = {path_tmp[i][1][0], path_tmp[i][1][1]};
            if((l1[0] >= p1[0] && l1[0] <= p2[0] && l1[1] >= p1[1] && l1[1] <= p2[1]) || 
                (l2[0] >= p1[0] && l2[0] <= p2[0] && l2[1] >= p1[1] && l2[1] <= p2[1]))
                return true;
            if(segment(l1,l2,p1,p2) || segment(l1,l2,p3,p4))
                return true;
        }
        return false;
    }

    void Robot::ChangePath()
    {
        vector<float> EndPos = g_planner.last_endposition;
        set_goal(EndPos[0]*100, 448-EndPos[1]*100);
    }
}