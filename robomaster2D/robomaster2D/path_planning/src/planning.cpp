#include "planning.hpp"

namespace critical_hit_planning
{
    Planning::Planning()
    {
        nav_state_ = false;
        find_path_flag_ = false;
        IcraMap_.read_map_param("./icra2020.pgm");
        set_map();
    }

    Planning::~Planning()
    {
        
    }

    void Planning::set_goal(double goal_x, double goal_y)
    {
        goal[0] = goal_x;
        goal[1] = goal_y;
        nav_state_ = true;     //规划标志为true
        find_path_flag_ = false;//规划完成标志为false
    }

    void Planning::reset_nav()
    {
        find_path_flag_ = false;//规划完成标志为false
    }

    bool Planning::path_plan(double &vx, double &vy, double &angular, double &path_angular, double pos_x, double pos_y, double pos_angle)
    {
        current_x = pos_x;
        current_y = pos_y;
        current_angle = pos_angle;
        bool is_block_in_path = false;
        if(nav_state_ == true)//开始规划
        {
            if(find_path_flag_ == false)
            {
                if(!Check_Goal(pos_x, pos_y)){
                    nav_state_ = false;
                    return true;
                }
                astar_path_ = astar_.search_path(IcraMap_.costmap_, pos_x*20, pos_y*20, goal[0]*20, goal[1]*20);
                if(astar_path_.size() > 3)
                {
                    find_path_flag_ = true;
                    smooth_sharp_corner_.set_delta_max(0.05);
                    smooth_path_ = smooth_sharp_corner_.smooth_sharp_corner(IcraMap_.costmap_,astar_path_);
                    final_path_ = look_ahead_.update_velocity(std::sqrt(vx*vx+vy*vy), smooth_path_);
                    navigation_.reset(pos_angle);
                }
            }
            is_block_in_path = Check_Path(navigation_.current_target_id_);
            if(find_path_flag_ == false || is_block_in_path)
            {
                nav_state_ = false;
                vx = 0;
                vy = 0;
                angular = 0;
            }
            else
            {
                if(final_path_.size()){
                    if(!navigation_.navigation_path(pos_x, pos_y, pos_angle, final_path_, vx, vy, angular, path_angular))
                    {
                        nav_state_ = false;
                        vx = 0;
                        vy = 0;
                        angular = 0;
                    }
                }
            }
            if(show_map){
                show_visualization(pos_x, pos_y, pos_angle);
            }
        }
        return is_block_in_path;

    }

    void Planning::clean_block() {
        IcraMap_.clean_map_block();
        set_map();
        reset_nav();
    }
    void Planning::add_block(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4){
        cv::Point pts[4];
        pts[0] = cv::Point(x1, y1);
        pts[1] = cv::Point(x2, y2);
        pts[2] = cv::Point(x3, y3);
        pts[3] = cv::Point(x4, y4);
        IcraMap_.add_map_block(pts);
        set_map(); //reset the map with new blocks
        reset_nav(); //reset the navigation path
//        cv::imshow("costmap", IcraMap_.costmap_);
//        if( cv::waitKey(100) == 1)
    }
    void Planning::set_map(){
        IcraMap_.set_map_param(49, 36.5, 5);
    }

    //鼠标监听回调函数
    void Planning::onMouse(int events, int x, int y, int, void* userdata)
    {
        Planning* temp = reinterpret_cast<Planning*>(userdata);
        temp->on_Mouse(events, x, y);//opencv指向静态函数
    }
    
    void Planning::on_Mouse(int event, int x, int y)
    {
        switch (event)
        {
            case cv::EVENT_LBUTTONDOWN://鼠标点击事件
            {
                set_goal((double)x/100, (double)(448-y)/100);//根据点击位置设置机器人的目标点坐标
            }break;
            case cv::EVENT_RBUTTONDOWN:
            {
                add_block(x/5, y/5, x/5+5, y/5, x/5+5, y/5+5, x/5, y/5+5);
            }break;
            case cv::EVENT_MBUTTONDOWN:
            {
                clean_block();
            }
                break;
            default:;
        }
    }

    bool Planning::Check_Path(int current_target_id_)
    {
        if(final_path_.size()){
            int pixel_x = 0;
            int pixel_y = 0;
            for(int i = 0;i<4; i++){
                pixel_x = final_path_[current_target_id_+i].x_*20;
                pixel_y = IcraMap_.costmap_.rows - final_path_[current_target_id_+i].y_*20;
                if(255 - IcraMap_.costmap_.at<unsigned char>(pixel_y, pixel_x) < 60) { // 判断目标点是否为障碍
                    return false;
                }
            }
            return true;
        }
    }
    bool Planning::Check_Goal(double start_x, double start_y)
    {
        float distance = sqrt((goal[0]-start_x)*(goal[0]-start_x) + (goal[1]-start_y)*(goal[1]-start_y));
        if(distance < 0.1) return false;// 若与起点过近则不进行规划，返回false
        int pixel_x = goal[0]*20;
        int pixel_y = IcraMap_.costmap_.rows - goal[1]*20;
        int point_num = 8;

        if(255 - IcraMap_.costmap_.at<unsigned char>(pixel_y, pixel_x) > 20){ // 判断目标点是否为障碍
//            std::cout << "Goal is in obstacle! " << std::endl;
            double r = 0.5;
            bool get_new_goal = false;
            // 寻找以目标点为圆心的路径点作为新的目标
            while(r < 5)
            {
                double new_goal_x = 0,new_goal_y = 0,goal_x = 0,goal_y = 0;distance = 100000.0f;
                for(int i = 0; i < point_num; i++)
                {
                    new_goal_x = goal[0] + r*sin(CV_PI*2*i/point_num);
                    new_goal_y = goal[1] + r*cos(CV_PI*2*i/point_num);
                    new_goal_x = new_goal_x<0.25?0.25:new_goal_x;
                    new_goal_x = new_goal_x>7.75?7.75:new_goal_x;
                    new_goal_y = new_goal_y<0.25?0.25:new_goal_y;
                    new_goal_y = new_goal_y>4.23?4.23:new_goal_y;
                    if(IcraMap_.costmap_.at<unsigned char>(IcraMap_.costmap_.rows-new_goal_y*20, new_goal_x*20) >= 255)
                    { // 若新目标点无障碍
                        get_new_goal = true;
                        double new_distance = sqrt((new_goal_x-start_x)*(new_goal_x-start_x) + (new_goal_y-start_y)*(new_goal_y-start_y));
                        if(new_distance < distance){ // 选择与起点距离最小的点作为新的目标点
                            goal_x = new_goal_x;
                            goal_y = new_goal_y;
                            distance = new_distance;
                        }
                    }
                }
                if(!get_new_goal)
                {
                    r += 0.05;
                } else{
                    goal[0] = goal_x;
                    goal[1] = goal_y;
                    return true;
                }
            }
            std::cout << "Fail to get new Goal! " << std::endl;
            return false;
        }
        return true;
    }

    bool Planning::Is_Nav()
    {
        if(nav_state_)
            return true;
        else return false;
    }


    void Planning::show_visualization(double pos_x, double pos_y, double pos_angle)//A星地图可视化
    {
        cv::resize(IcraMap_.costmap_, icra_map_, cv::Size(815, 450));
        cv::cvtColor(icra_map_, icra_map_, CV_GRAY2BGR);
//        printf("!");
        // for(int i = 0; i < astar_path_.size(); i++)
        // {
        //     cv::circle(icra_map_, cv::Point(astar_path_[i].x*100, 448-astar_path_[i].y*100), 2, cv::Scalar(0,255,0),2);
        // }
        for(int i = 0; i < final_path_.size(); i++)
        {
            if(i == navigation_.current_id_)
            {
                cv::circle(icra_map_, cv::Point(final_path_[i].x_*100, 448-final_path_[i].y_*100), 2,
                        cv::Scalar(0,0,255),2);
            }
            else if(i == navigation_.current_target_id_)
            {
                cv::circle(icra_map_, cv::Point(final_path_[i].x_*100, 448-final_path_[i].y_*100), 2,
                        cv::Scalar(255,0,0),2);
            }
            else
            {
                cv::circle(icra_map_, cv::Point(final_path_[i].x_*100, 448-final_path_[i].y_*100), 2,
                            cv::Scalar(0,255,255),2);
            }
        }

        char str[10];
        //sprintf(str, "speed: %f m/s", vx);
        cv::putText(icra_map_, str, cv::Point(0,30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
        //sprintf(str, "angular: %f rad/s", angular);
        cv::putText(icra_map_, str, cv::Point(200,30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
        cv::arrowedLine(icra_map_, cv::Point(pos_x*100, icra_map_.rows-pos_y*100),
                        cv::Point(pos_x*100+20*cos(pos_angle), icra_map_.rows-pos_y*100-20*sin(pos_angle)),
                        cv::Scalar(255,0,0), 2);
        cv::imshow("plan_debug", icra_map_);
        if(cv::waitKey(1)==1) show_map = false;
    }

    void Planning::get_block_map(int (*out_map)[81])
    {
        IcraMap_.get_block_map(out_map);
    }
} // namespace critical_hit_planning
