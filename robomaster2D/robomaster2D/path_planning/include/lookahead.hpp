#ifndef CRITICAL_HIT_LOOKAHEAD_H
#define CRITICAL_HIT_LOOKAHEAD_H

#include "smooth.hpp"

namespace critical_hit_planning
{
    class LookAhead
    {
        public:
        LookAhead();
        std::vector<path_point> update_velocity(double current_v, std::vector<smooth_path> path);
        private:
        double v_max_;
        double a_max_;
        double omiga_max_;
        double trapezoidal_plan(double x_start, double x_end, double x_max, double acc, double d, double l);
    };
}

#endif