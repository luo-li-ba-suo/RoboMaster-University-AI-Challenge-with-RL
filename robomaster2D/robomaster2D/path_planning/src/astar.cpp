#include "astar.hpp"
#include "planning.hpp"
#include <math.h>
#include <stddef.h>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>


namespace critical_hit_planning
{
    struct Compare {
        bool operator()(const list_node& a, const list_node& b) {
            return a.cost_ >= b.cost_;
        }
    };
    void AStar::InitMap(cv::Mat costmap)
    {
        width_ = costmap.cols;
        height_ = costmap.rows;
        nodes_.resize(costmap.rows);
        for(int i = 0; i < nodes_.size(); i++)
        {
            nodes_[i].resize(costmap.cols);
            for(int j = 0; j < nodes_[i].size(); j++)
            {
                nodes_[i][j].parent = {-1,-1};
                nodes_[i][j].f = 0;
                nodes_[i][j].g = 0;
                nodes_[i][j].h = 0;
                nodes_[i][j].ex_cost = (255 - costmap.at<unsigned char>(i, j));
                nodes_[i][j].is_in_close_list_ = false;
                nodes_[i][j].is_in_open_list_ = false;
            }
        }
    }

    std::vector<struct Point> AStar::search_path(cv::Mat costmap, int start_x, int start_y, int goal_x, int goal_y)
    {
        InitMap(costmap);
        start_y = costmap.rows - 1 - start_y;
        goal_y = costmap.rows - 1 - goal_y;
        std::vector<struct Point> path;
        if((goal_x == start_x) && (goal_y == start_y))
        {
            return path;
        }
        // 初始化open_set
        std::priority_queue<list_node, std::vector<list_node>, Compare> open_list_;
        // 将起点加入open_set中
        open_list_.push(list_node(start_x, start_y, nodes_[start_y][start_x].f));
        nodes_[start_y][start_x].is_in_open_list_ = true;
        while(true)
        {
            // 将节点n加入close_set中
            int current_node_x = open_list_.top().x_;
            int current_node_y = open_list_.top().y_;
            nodes_[current_node_y][current_node_x].is_in_close_list_ = true;
            // 如果节点n为终点
            if(current_node_x == goal_x && current_node_y == goal_y)
            {
                break;
            }
            // 如果节点n不是终点，则
            // 将节点n从open_set中删除
            open_list_.pop();
            // 遍历节点n所有的邻近节点
            std::vector<std::vector<int>> neighbors = getNeighbors(current_node_x, current_node_y);
            for(int i = 0; i < neighbors.size(); i++)
            {
                // 计算节点m的优先级
                double delta_x = neighbors[i][0] - current_node_x;
                double delta_y = neighbors[i][1] - current_node_y;
                int parent_x = nodes_[current_node_y][current_node_x].parent[0];
                int parent_y = nodes_[current_node_y][current_node_x].parent[1];
                double step_cost = std::sqrt(delta_x*delta_x+delta_y*delta_y);
                double cost_g = step_cost + nodes_[current_node_y][current_node_x].g + 
                                nodes_[current_node_y][current_node_x].ex_cost;
                if(parent_x != -1 && parent_y != -1)
                {
                    double v_x = current_node_x - parent_x;
                    double v_y = current_node_y - parent_y;
                    double last_step_cost = std::sqrt(v_x*v_x+v_y*v_y);
                    double cross_product = (delta_x/step_cost) * (v_y/last_step_cost) - 
                                            (v_x/last_step_cost) * (delta_y/step_cost);
                    if(fabs(cross_product) > 1e-5) cost_g += 2;
                }
                if(nodes_[neighbors[i][1]][neighbors[i][0]].is_in_open_list_ == true)
                {
                    if(cost_g > nodes_[neighbors[i][1]][neighbors[i][0]].g)
                    {
                        continue;
                    }
                    nodes_[neighbors[i][1]][neighbors[i][0]].g = cost_g;
                
                    // 设置节点m的parent为节点n
                    nodes_[neighbors[i][1]][neighbors[i][0]].parent = {current_node_x, current_node_y};
                    
                    nodes_[neighbors[i][1]][neighbors[i][0]].f = nodes_[neighbors[i][1]][neighbors[i][0]].g + 
                                                                    nodes_[neighbors[i][1]][neighbors[i][0]].h;
                }
                else
                {
                    delta_x = fabs(goal_x - neighbors[i][0]);
                    delta_y = fabs(goal_y - neighbors[i][1]);
                    double min_delta = std::min(delta_x, delta_y);
                    double max_delta = std::max(delta_x, delta_y);
                    nodes_[neighbors[i][1]][neighbors[i][0]].h = 0;//min_delta * std::sqrt(2) + max_delta - min_delta - max_delta * 0.1;
                    // if(min_delta != 0 && min_delta != max_delta)
                    // {
                    //     nodes_[neighbors[i][1]][neighbors[i][0]].h += 0.707;
                    // }
                    nodes_[neighbors[i][1]][neighbors[i][0]].g = cost_g;
                    
                    // 设置节点m的parent为节点n
                    nodes_[neighbors[i][1]][neighbors[i][0]].parent = {current_node_x, current_node_y};
                    
                    nodes_[neighbors[i][1]][neighbors[i][0]].f = nodes_[neighbors[i][1]][neighbors[i][0]].g + 
                                                                    nodes_[neighbors[i][1]][neighbors[i][0]].h;
                    // 将节点m加入open_set中
                    open_list_.push(list_node(neighbors[i][0], neighbors[i][1], nodes_[neighbors[i][1]][neighbors[i][0]].f));
                    nodes_[neighbors[i][1]][neighbors[i][0]].is_in_open_list_ = true;
                }
            }
        }
        int current_x = goal_x;
        int current_y = goal_y;
        while(true)
        {
            struct Point p;
            p.x = (double)current_x / 20;
            p.y = (double)(height_ - 1 - current_y) / 20;
            path.push_back(p);
            int parent_x = nodes_[current_y][current_x].parent[0];
            int parent_y = nodes_[current_y][current_x].parent[1];
            current_x = parent_x;
            current_y = parent_y;
            if(parent_x == start_x && parent_y == start_y)
            {
                struct Point p1;
                p1.x = (double)parent_x / 20;
                p1.y = (double)(height_ - 1 - parent_y) / 20;
                path.push_back(p1);
                break;
            }
        }
        if(path.size() > 1)
            std::reverse(path.begin(), path.end());
        return path;
    }

    std::vector<std::vector<int>> AStar::getNeighbors(int x, int y)
    {
        std::vector<std::vector<int>> neighbors;
        if(x > 0 && y > 0)
        {
            if(nodes_[y-1][x-1].is_in_close_list_ == false)
                neighbors.push_back({x-1, y-1});
        }
        if(x > 0)
        {
            if(nodes_[y][x-1].is_in_close_list_ == false)
                neighbors.push_back({x-1, y});
        }
        if(x > 0 && y < height_-1)
        {
            if(nodes_[y+1][x-1].is_in_close_list_ == false)
                neighbors.push_back({x-1, y+1});
        }
        if(y < height_ - 1)
        {
            if(nodes_[y+1][x].is_in_close_list_ == false)
                neighbors.push_back({x, y+1});
        }
        if(x < width_ - 1 && y < height_ - 1)
        {
            if(nodes_[y+1][x+1].is_in_close_list_ == false)
                neighbors.push_back({x+1, y+1});
        }
        if(x < width_ - 1)
        {
            if(nodes_[y][x+1].is_in_close_list_ == false)
                neighbors.push_back({x+1, y});
        }
        if(x < width_ - 1 && y > 0)
        {
            if(nodes_[y-1][x+1].is_in_close_list_ == false)
                neighbors.push_back({x+1, y-1});
        }
        if(y > 0)
        {
            if(nodes_[y-1][x].is_in_close_list_ == false)
                neighbors.push_back({x, y-1});
        }
        return neighbors;
    }
}
