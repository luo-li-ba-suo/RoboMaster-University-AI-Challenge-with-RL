#include "build_map.hpp"
#include <math.h>
#include <algorithm>

namespace critical_hit_planning
{
	void Map::update_costs()
	{
		costmap_ = map_img_.clone();
		if(cell_inflation_radius_ == 0)
			return;
		unsigned int size_x = map_img_.cols, size_y = map_img_.rows;

		if (seen_ == NULL) {
			seen_size_ = size_x * size_y;
			seen_ = new bool[seen_size_];
		}
		memset(seen_, false, size_x * size_y * sizeof(bool));

		// Inflation list; we append cells to visit in a list associated with its distance to the nearest obstacle
		// We use a map<distance, list> to emulate the priority queue used before, with a notable performance boost

		// Start with lethal obstacles: by definition distance is 0.0
		std::vector<CellData> &obs_bin = inflation_cells_[0.0];
		for (int j = 0; j < costmap_.rows; j++) {
			for (int i = 0; i < costmap_.cols; i++) {
				int index = i + j * costmap_.cols;
				unsigned char cost = 255-costmap_.at<unsigned char>(j, i);
				if (cost >= 100) {
					obs_bin.push_back(CellData(index, i, j, i, j));
				}
			}
		}

		// Process cells by increasing distance; new cells are appended to the corresponding distance bin, so they
		// can overtake previously inserted but farther away cells
		std::map<double, std::vector<CellData> >::iterator bin;
		for (bin = inflation_cells_.begin(); bin != inflation_cells_.end(); ++bin) {
			for (int i = 0; i < bin->second.size(); ++i) {
			// process all cells at distance dist_bin.first
			const CellData &cell = bin->second[i];

			unsigned int index = cell.index_;

			// ignore if already visited
			if (seen_[index]) {
				continue;
			}

			seen_[index] = true;

			unsigned int mx = cell.x_;
			unsigned int my = cell.y_;
			unsigned int sx = cell.src_x_;
			unsigned int sy = cell.src_y_;

			// assign the cost associated with the distance from an obstacle to the cell
			unsigned char cost = CostLookup(mx, my, sx, sy);
			unsigned char old_cost = 255-costmap_.at<unsigned char>(my, mx);
			// 膨胀操作，代价值通过数组索引获得
			costmap_.at<unsigned char>(my, mx) = 255-std::max(old_cost, cost);

			// attempt to put the neighbors of the current cell onto the inflation list
			// 通过改变该栅格的上下左右四个栅格来实现膨胀
			if (mx > 0)
				Enqueue(index - 1, mx - 1, my, sx, sy);
			if (my > 0)
				Enqueue(index - size_x, mx, my - 1, sx, sy);
			if (mx < size_x - 1)
				Enqueue(index + 1, mx + 1, my, sx, sy);
			if (my < size_y - 1)
				Enqueue(index + size_x, mx, my + 1, sx, sy);
			}
		}

		inflation_cells_.clear();
	}

	// 计算障碍物膨胀区域的代价，存储在cached_costs_里面
	void Map::ComputeCaches() {
		if (cell_inflation_radius_ == 0)
			return;

		// based on the inflation radius... compute distance and cost caches
		// 如果膨胀半径变化
        if(cached_costs_!=nullptr){
            for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
                delete []cached_costs_[i];
            }
            delete []cached_costs_;
        }

        if(cached_distances_!=nullptr){
            for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
                delete []cached_distances_[i];
            }
            delete []cached_distances_;
        }

        cached_costs_ = new unsigned char *[cell_inflation_radius_ + 2];
		cached_distances_ = new double *[cell_inflation_radius_ + 2];

		for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
			cached_costs_[i] = new unsigned char[cell_inflation_radius_ + 2];
			cached_distances_[i] = new double[cell_inflation_radius_ + 2];
			for (unsigned int j = 0; j <= cell_inflation_radius_ + 1; ++j) {
				cached_distances_[i][j] = hypot(i, j);
			}
		}

		for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
			for (unsigned int j = 0; j <= cell_inflation_radius_ + 1; ++j) {
				cached_costs_[i][j] = ComputeCost(cached_distances_[i][j]);
			}
		}
		// for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
		// 	std::cout << cached_distances_[i][i] << ":" << int(cached_costs_[i][i]) << std::endl;
		// }
	}

	/**
	 * @brief  Given an index of a cell in the costmap, place it into a list pending for obstacle inflation
	 * @param  grid The costmap
	 * @param  index The index of the cell
	 * @param  mx The x coordinate of the cell (can be computed from the index, but saves time to store it)
	 * @param  my The y coordinate of the cell (can be computed from the index, but saves time to store it)
	 * @param  src_x The x index of the obstacle point inflation started at
	 * @param  src_y The y index of the obstacle point inflation started at
	 */
	inline void Map::Enqueue(unsigned int index, unsigned int mx, unsigned int my,
										unsigned int src_x, unsigned int src_y) {
		if (!seen_[index]) {
			// we compute our distance table one cell further than the inflation radius dictates so we can make the check below
			double distance = DistanceLookup(mx, my, src_x, src_y);

			// we only want to put the cell in the list if it is within the inflation radius of the obstacle point
			if (distance > cell_inflation_radius_)
				return;

			// push the cell data onto the inflation list and mark
			inflation_cells_[distance].push_back(CellData(index, mx, my, src_x, src_y));
		}
	}

    Map::~Map() {
        delete [] seen_;
        if(cached_costs_!=nullptr){
            for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
                delete []cached_costs_[i];
            }
            delete []cached_costs_;
        }

        if(cached_distances_!=nullptr){
            for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i) {
                delete []cached_distances_[i];
            }
            delete []cached_distances_;
        }
    }
}
