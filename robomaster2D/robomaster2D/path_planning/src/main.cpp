#include <opencv2/opencv.hpp>
#include "planning.hpp"

using namespace critical_hit_planning;

Planning path_plan;
bool show_scene = true;
bool launch_mouse_guide = true;
double robot_x = 50;
double robot_y = 50;
double robot_angle = CV_PI/2;

cv::Mat set_robot_to(cv::Mat map, int x, int y, double angle)
{
    cv::Mat map_copy = map.clone();
    cv::RotatedRect rRect = cv::RotatedRect(cv::Point(x,map_copy.rows-y),cv::Size2f(60,45),-angle*180/CV_PI);
    cv::Point2f vertices[4];
    rRect.points(vertices);//提取旋转矩形的四个角点
    for(int i=0;i<4;i++)
    {
        cv::line(map_copy,vertices[i],vertices[(i+1)%4],cv::Scalar(255,0,0), 2);//四个角点连成线，最终形成旋转的矩形。
    }
    return map_copy;
}

double delta_time = 0.006;
void move_robot(double &robot_x, double &robot_y, double &robot_angle, double path_angle, double vx, double vy, double angular)
{
    vx *= 100; vy *= 100;
    robot_x += (vx * cos(path_angle) - vy * sin(path_angle)) * delta_time;
    robot_y += (vx * sin(path_angle) + vy * cos(path_angle)) * delta_time;
    robot_angle += angular * delta_time;
    if(robot_angle > CV_PI) robot_angle -= CV_PI*2;
    if(robot_angle < -CV_PI) robot_angle += CV_PI*2;
}

int main()
{
    if (launch_mouse_guide){
        path_plan.show_map = true;
    } else{
        path_plan.show_map = false;
    }
    cv::Mat map = cv::imread("../icra2020.pgm");
    cv::resize(map, map, cv::Size(815,450));
    double vx = 0, vy = 0, angular = 0, path_angular=0;
    const char* winname = "robot_planning";
    while(cv::waitKey(1) != 27)
    {
        std::clock_t start = std::clock();
        path_plan.path_plan(vx, vy, angular, path_angular, robot_x/100, robot_y/100, robot_angle);
        move_robot(robot_x, robot_y, robot_angle, path_angular, vx, vy, angular);
        if(show_scene)
        {
            cv::Mat show_map = set_robot_to(map, robot_x, robot_y, robot_angle);
            cv::imshow(winname, show_map);
        }
        std::clock_t end = std::clock();
        delta_time = (double)(end-start)/CLOCKS_PER_SEC;
    }
    return 0;
}