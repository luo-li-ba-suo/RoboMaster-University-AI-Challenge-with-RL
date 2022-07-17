#ifndef CRITICAL_HIT_BUILD_MAP
#define CRITICAL_HIT_BUILD_MAP

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace critical_hit_planning
{
    struct Point
    {
        double x;
        double y;
        double v;
    };
    /**
     * @class CellData
     * @brief Storage for cell information used during obstacle inflation
     */
    class CellData {
        public:
        /**
         * @brief  Constructor for a CellData objects
         * @param  i The index of the cell in the cost map
         * @param  x The x coordinate of the cell in the cost map
         * @param  y The y coordinate of the cell in the cost map
         * @param  sx The x coordinate of the closest obstacle cell in the costmap
         * @param  sy The y coordinate of the closest obstacle cell in the costmap
         * @return
         */
        CellData(double i, unsigned int x, unsigned int y, unsigned int sx, unsigned int sy) :
            index_(i), x_(x), y_(y), src_x_(sx), src_y_(sy) {
        }
        unsigned int index_;
        unsigned int x_, y_;
        unsigned int src_x_, src_y_;
    };
    class Map
    {
        public:
        Map():
         seen_(NULL),
         cached_costs_(NULL),
         cached_distances_(NULL)
        {
        }
        void set_map_param(double inflation_radius, double inscribed_radius, double resolution)
        {
            inflation_radius_ = inflation_radius;
            resolution_ = resolution;
            inscribed_radius_ = inscribed_radius;
            cell_inflation_radius_ = inflation_radius_ / resolution_;
            ComputeCaches();
            update_costs();
        }
        void read_map_param(std::string map_name)
        {
            map_img_original = cv::imread(map_name, cv::IMREAD_GRAYSCALE);
            if(map_img_original.empty()) std::cout << "[ERROR] Do NOT get the image of map!" << std::endl;
            map_img_ = map_img_original.clone();
        }
        void clean_map_block()
        {
            map_img_ = map_img_original.clone();
        }
        void add_map_block(cv::Point points[])
        {
            const cv::Point* ppt[1] = { points };
            int npt[] = {4};
            cv::fillPoly(map_img_, ppt, npt, 1, cv::Scalar(0));
        }
        void get_block_map(int (*out_map)[81]) const
        {
            cv::Mat map_img_resized;
            cv::resize(map_img_, map_img_resized, cv::Size(81, 45));
            for(int i=0;i<map_img_resized.rows;++i)
            {
                for(int j=0;j<map_img_resized.cols;++j)
                {
                    out_map[i][j] = int(map_img_resized.ptr<uchar>(i)[j]);
                }
            }
        }
        cv::Mat map_img_;
        cv::Mat map_img_original;
        cv::Mat costmap_;
        void update_costs();
        private:
        double resolution_;
        double inflation_radius_;
        double inscribed_radius_;
        int cell_inflation_radius_;
        std::map<double, std::vector<CellData> > inflation_cells_;
        unsigned char** cached_costs_;
        double** cached_distances_;
        bool* seen_;
        int seen_size_;

        /** @brief  Given a distance, compute a cost.
         * @param  distance The distance from an obstacle in cells
         * @return A cost value for the distance */
        inline unsigned char ComputeCost(unsigned distance_cell) const {
            unsigned char cost = 0;
            // 0代表障碍物边缘
            if (distance_cell == 0)
                cost = 255;
            else if (distance_cell * resolution_ <= inscribed_radius_)
                cost = 254;
            else {
                // make sure cost falls off by Euclidean distance
                double euclidean_distance = distance_cell * resolution_;
                // 因为这里用到了机器人足迹的内接圆半径inscribed_radius_，所以需要在机器人足迹发生变化时重新调用该函数
                double factor = exp(-1.0 * (euclidean_distance - inscribed_radius_));
                cost = (unsigned char) (253 * factor);
            }
            return cost;
        }
        void ComputeCaches();
        /**
         * @brief  Lookup pre-computed distances
         * @param mx The x coordinate of the current cell
         * @param my The y coordinate of the current cell
         * @param src_x The x coordinate of the source cell
         * @param src_y The y coordinate of the source cell
         * @return
         */
        inline double DistanceLookup(int mx, int my, int src_x, int src_y) {
            unsigned int dx = abs(mx - src_x);
            unsigned int dy = abs(my - src_y);
            return cached_distances_[dx][dy];
        }

        /**
         * @brief  Lookup pre-computed costs
         * @param mx The x coordinate of the current cell
         * @param my The y coordinate of the current cell
         * @param src_x The x coordinate of the source cell
         * @param src_y The y coordinate of the source cell
         * @return
         */
        inline unsigned char CostLookup(int mx, int my, int src_x, int src_y) {
            unsigned int dx = abs(mx - src_x);
            unsigned int dy = abs(my - src_y);
            return cached_costs_[dx][dy];
        }
        inline void Enqueue(unsigned int index, unsigned int mx, unsigned int my,
                            unsigned int src_x, unsigned int src_y);
    };
}

#endif
