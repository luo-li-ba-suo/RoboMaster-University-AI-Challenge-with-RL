#ifndef CRITICAL_HIT_ASTAR_H
#define CRITICAL_HIT_ASTAR_H

#include "build_map.hpp"

namespace critical_hit_planning
{
    struct AStarNode
    {
        std::vector<int> parent;
        int node_id;
        double f;
        double h;
        double g;
        int ex_cost;
        bool is_in_close_list_;
        bool is_in_open_list_;
    };
    class list_node
    {
        public:
        list_node(int x, int y, double cost)
        {
            x_ = x;
            y_ = y;
            cost_ = cost;
        }
        int x_;
        int y_;
        double cost_;
    };
    class AStar
    {
        public:
        void InitMap(cv::Mat costmap);
        std::vector<struct Point> search_path(cv::Mat costmap, int start_x, int start_y, int goal_x, int goal_y);
        private:
        std::vector<std::vector<struct AStarNode>> nodes_;
        std::vector<std::vector<int>> getNeighbors(int x, int y);
        int width_;
        int height_;
    };
}

#endif