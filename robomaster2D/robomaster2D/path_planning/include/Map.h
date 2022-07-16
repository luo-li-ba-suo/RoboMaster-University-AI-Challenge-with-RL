#pragma once
#ifndef ICRA_MAP_H
#define ICRA_MAP_H

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
using std::vector;
namespace hitcrt{
	using uint = unsigned int;
	//自定义地图类
	class Map
	{
	public:
		//            导航节点的x，           导航节点的y
		vector<float> navigation_positionx, navigation_positiony;
		//                    静态障碍物的中心
		vector<vector<float>> obstacles_center;
		//            静态障碍物的x偏移量， 静态障碍物的y偏移量
		vector<float> obstacles_offsetx, obstacles_offsety;
		Map()
		{
			navigation_positionx = {0.5, 1.25, 1.9, 2.95, 4.04, 5.13, 6.18, 6.83, 7.58};
			navigation_positiony = {0.445, 1.65, 2.83, 4.035};
            obstacles_center = {
            {0.5, 3.38}, {1.9, 2.24}, {1.6, 0.5}, {4.04, 3.445},
            {4.04, 2.24},
            {4.04, 1.035}, {6.48, 3.98}, {6.18, 2.24}, {7.58, 1.1}};
            obstacles_offsetx = {0.5, 0.4, 0.1, 0.5, 0.177, 0.5, 0.1, 0.4, 0.5};
            obstacles_offsety = {0.1, 0.1, 0.5, 0.1, 0.177, 0.1, 0.5, 0.1, 0.1};
		}
	};

}
#endif