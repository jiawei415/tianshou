import numpy as np
import gym
from gym import spaces
from typing import Optional


class CustomizeMDPV1(gym.Env):
    def __init__(self, length: int = 5, seed: Optional[int] = None, reward_threshold=0.5):
        super().__init__()
        self._length = length
        self._rng = np.random.RandomState(seed)
        self._final_reward = self._rng.randint(2)
        self._remaining_reward= self._rng.uniform(0, reward_threshold, size=length-1)

        self._row = 0
        self._column = 0

    @property
    def observation_space(self):
        return spaces.Discrete(self._length* 2 - 1)

    @property
    def action_space(self):
        return spaces.Discrete(2)

    def _get_observation(self):
        obs = np.zeros(shape=(2, self._length), dtype=np.float32)
        obs[self._row, self._column] = 1.
        return obs.flatten()[:self.observation_space.n]

    def reset(self):
        self._row = 0
        self._column = 0
        return self._get_observation()

    def step(self, action):
        if action == 0:
            reward = (1- self._row) * self._remaining_reward[self._column]
            self._column += self._row * 1
            self._row = 1
        elif action == 1:
            self._column += 1
            reward = (self._column == self._length - 1) * self._final_reward
        else:
            raise ValueError(f"illegal action {action}")
        
        done = (self._row == 0 and self._column == self._length - 1) or \
               (self._row == 1 and self._column == self._length - 2)
        obs = self._get_observation()
        return obs, reward, done, {}

    def render(self):
        print(self.__repr__())

    def __repr__(self):
        # obs = np.zeros(shape=(2, self._length), dtype=np.float32)
        # obs[1][-1] = 2
        # obs[self._row, self._column] = 1.
        # return obs
        s = []
        position = [self._row, self._column]
        end = [1, self._length - 1]
        for i in range(2):
            for j in range(self._length):
                if [i, j] == position:
                    s.append('@')
                elif [i, j] == end:
                    s.append('$')
                else:
                    s.append('.')
            s.append('\n')
        return ''.join(s)


class CustomizeMDPV2(gym.Env):
    def __init__(self, length: int = 5, seed: Optional[int] = None):
        super().__init__()
        self._length = length
        self._rng = np.random.RandomState(seed)
        reward_index = self._rng.randint(self._length-1)
        self._all_rewards = np.zeros(self._length-1)
        self._all_rewards[reward_index] = 1.

        self._row = 0
        self._column = 0

    @property
    def observation_space(self):
        return spaces.Discrete(self._length* 2 - 1)

    @property
    def action_space(self):
        return spaces.Discrete(2)

    def _get_observation(self):
        obs = np.zeros(shape=(2, self._length), dtype=np.float32)
        obs[self._row, self._column] = 1.
        return obs.flatten()[:self.observation_space.n]

    def reset(self):
        self._row = 0
        self._column = 0
        return self._get_observation()

    def step(self, action):
        if action == 0:
            reward = (1- self._row) * self._all_rewards[self._column]
            self._column += self._row * 1
            self._row = 1
        elif action == 1:
            self._column += 1
            reward = 0.
        else:
            raise ValueError(f"illegal action {action}")
        
        done = (self._row == 0 and self._column == self._length - 1) or \
               (self._row == 1 and self._column == self._length - 2)
        obs = self._get_observation()
        return obs, reward, done, {}

    def render(self):
        print(self.__repr__())

    def __repr__(self):
        # obs = np.zeros(shape=(2, self._length), dtype=np.float32)
        # obs[1][-1] = 2
        # obs[self._row, self._column] = 1.
        # return obs
        s = []
        position = [self._row, self._column]
        end = [1, self._length - 1]
        for i in range(2):
            for j in range(self._length):
                if [i, j] == position:
                    s.append('@')
                elif [i, j] == end:
                    s.append('$')
                else:
                    s.append('.')
            s.append('\n')
        return ''.join(s)


if __name__ == "__main__":
    env = CustomizeMDPV1(length=20)
    # env = CustomizeMDPV2(length=10)
    action_space = env.action_space
    observation_space = env.observation_space
    print(f"action_sapce: {action_space} obs_space: {observation_space}")
    obs = env.reset()
    print(f"obs: {obs}")
    env.render()
    total_reward = 0.
    while True:
        act = action_space.sample()
        obs, reward, done, info = env.step(act)
        print(f"act:{act} obs: {obs}")
        env.render()
        total_reward += reward
        if done:
            print(f"total_reward: {total_reward}")
            break
