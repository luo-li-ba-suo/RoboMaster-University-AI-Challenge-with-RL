import numpy as np
import copy

class ModelPool:
    def __init__(self, capacity, self_play_mode, delta_historySP):
        self.capacity = capacity
        self.pool = []
        self.pool_step = []
        self.self_play_mode = self_play_mode
        self.delta = delta_historySP
        self.step = 0
        self.delete_step = 0
        self.model_num = 0

    def push_model(self, model):
        self.step += 1
        self.pool.append(copy.deepcopy(model))
        self.pool_step.append(self.step)
        if self.model_num < self.capacity:
            self.model_num += 1
        if self.self_play_mode == 1:
            # delta self play mode
            current_num = len(self.pool)
            if current_num > self.capacity:
                if self.step * self.delta < self.capacity:
                    del self.pool[0]
                    del self.pool_step[0]
                else:
                    if self.pool_step[0] < self.step * (1 - self.delta):
                        del self.pool[0]
                        del self.pool_step[0]
                    else:
                        del self.pool[self.delete_step]
                        del self.pool_step[self.delete_step]
                        self.delete_step += 1
                        if self.delete_step >= self.capacity:
                            self.delete_step = 0

    def pull_model(self):
        random_id = np.random.randint(0, self.model_num)
        return self.pool[random_id]
