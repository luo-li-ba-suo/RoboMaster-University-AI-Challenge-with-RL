#ifndef CRITICAL_HIT_PLANNING
#define CRITICAL_HIT_PLANNING

#include <opencv2/opencv.hpp>
#include <thread>
#include "build_map.hpp"
#include "astar.hpp"
#include "smooth.hpp"
#include "lookahead.hpp"
#include "navigation.hpp"

namespace critical_hit_planning
{
    class Planning
    {
        public:
        double current_x=0;
        double current_y=0;
        double current_angle=0;
        Planning();
        ~Planning();
        void show_visualization(double pos_x, double pos_y, double pos_angle);
        void set_goal(double goal_x, double goal_y);
        void reset_nav();
        bool Check_Goal(double goal_x, double goal_y);
        bool Check_Path(int current_target_id_);
        bool Is_Nav();
        bool path_plan(double &vx, double &vy, double &angular, double &path_angular, double pos_x, double pos_y, double pos_angle);
        void clean_block();
        void add_block(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4);
        void set_map();
        bool show_map = true;

        // Added by Kideng
        void get_block_map(int (*out_map)[81]);

        private:
        cv::Mat icra_map_;
        Map IcraMap_;
        void on_Mouse(int events, int x, int y);
        static void onMouse(int events, int x, int y, int, void* userdata);
        bool nav_state_;
        double goal[2];
        AStar astar_;
        bool find_path_flag_;
        SmoothCorner smooth_sharp_corner_;
        std::vector<struct Point> astar_path_;
        std::vector<smooth_path> smooth_path_;
        LookAhead look_ahead_;
        std::vector<path_point> final_path_;
        Navigation navigation_;
    };
}

#endif