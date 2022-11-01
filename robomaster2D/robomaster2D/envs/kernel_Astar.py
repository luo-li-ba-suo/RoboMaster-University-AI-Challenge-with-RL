"""
A_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import heapq
import numpy as np


def search(goal, obs, s_start, bord, quiet=True, max_try_time=20):
    # search from end(s_start) to start(goal) by Astar
    # first, determine a valid start by widthwise and longitudinal searching
    offsets = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
    find_valid_goal = False
    new_goal = goal
    for i in range(20):
        offsets_cur = offsets * i
        for offset in offsets_cur:
            new_goal = tuple(goal + offset)
            if new_goal not in obs and bord[0] < new_goal[0] < bord[1] - 1 and bord[2] < new_goal[1] < bord[3] - 1:
                find_valid_goal = True
                break
            if i == 0:
                break
        if find_valid_goal:
            break
    if not find_valid_goal:
        if not quiet:
            print("determining a valid start failed")
        return [], None
    # second, determine a valid goal by broad prior search
    ply = 1
    current_offset = np.array([1, 1])
    diffusion_offset = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
    i = 0
    s_start_copy = s_start
    try_time = 0
    while ply < 11:
        if s_start_copy not in obs and bord[0] < s_start_copy[0] < bord[1]-1 and bord[2] < s_start_copy[1] < bord[3]-1:
            astar = AStar(s_start_copy, new_goal, "euclidean", obs)
            path, visited = astar.searching()
            try_time += 1
            if path:
                if try_time > 1 and not quiet:
                    print(f"tried {try_time} times")
                return path, visited
            else:
                if try_time > max_try_time:
                    if not quiet:
                        print(f"determining a valid goal failed, tried {try_time} times")
                    return [], None
        s_start_copy = tuple(current_offset+s_start)
        current_offset += diffusion_offset[i]
        if np.linalg.norm(current_offset, ord=1) == 2*ply:
            i += 1
            if i == 4:
                i = 0
                ply += 1
                current_offset += [1,1]
    if not quiet:
        print(f"determining a valid goal failed, tried {try_time} times")
    return [], None


class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, obs):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.u_set = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]  # feasible input set
        self.obs = obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                pre_s = self.PARENT[s]
                new_cost = self.g[s] + self.cost(s, s_n, pre_s)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal, pre_s):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        if pre_s[0] - s_start[0] == s_start[0] - s_goal[0] and pre_s[1] - s_start[1] == s_start[1] - s_goal[1]:
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
        else:
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1]) + 1

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal
        if s not in PARENT:
            return []
        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    astar = AStar(s_start, s_goal, "euclidean")
    plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
