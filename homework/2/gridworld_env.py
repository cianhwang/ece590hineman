import numpy as np
import gym
from gym import spaces
from random_agent import RandomAgent

class GridWorldEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, reward, total_traj = 0):
        super(GridWorldEnv, self).__init__()

        # Size of the 2D-grid
        self.reward = reward
        self.H, self.W = self.reward.shape
        # Initialize the agent at the right bottom of the grid
        self.agent_pos = np.array([self.H-1, self.W-1]).astype(np.float32)
        self.moves = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0])
        }
        self.traj = total_traj
        self.total_traj = total_traj

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have 4: left, right, up, down
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.H, self.W]), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        # Initialize the agent at the right bottom of the grid
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        self.agent_pos = np.array([self.H-1, self.W-1]).astype(np.float32)
        self.traj = self.total_traj

        return self.agent_pos

    def step(self, action):

        delta = self.moves[action]
        # IF REACH BOUNDARY: STAY STILL WITH REWARD = -1.
        # ELSE: MOVE TO NEW LOC AND RECEIVE REWARD
        if np.min(self.agent_pos + delta) < 0.0:
            rw = -1.0
        elif np.max(self.agent_pos + delta - np.array([self.H, self.W])) >= 0.0:
            rw = -1.0    
        else:
            # FOR SUTTONBARTO GRIDWORLD ESPECIALLY: MOVE TO A' AND B'
            x_c, y_c = self.agent_pos
            if x_c == 0 and y_c == 1:
                self.agent_pos += np.array([4, 0])
                rw = 10.0
            elif x_c == 0 and y_c == 3:
                self.agent_pos += np.array([2, 0])
                rw = 5.0
            else:
                self.agent_pos = self.agent_pos + delta
                rw = self.reward[int(self.agent_pos[0]), int(self.agent_pos[1])]
            
        self.traj = self.traj - 1
        done = bool(self.traj == 0)

        info = {}
        return self.agent_pos, rw, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        for i in range(self.H):
            for j in range(self.W):
                if self.agent_pos[0] == i and self.agent_pos[1] == j:
                    print('*', end = '')
                else:
                    print('.', end = '')
            print()

    def close(self):
        pass

def env_fn():
    my_reward = np.zeros((5, 5))
    
    env = GridWorldEnv(my_reward, 0)
    env.seed(38)
    
    return env
    
# test env:
if __name__ == '__main__':

    env = env_fn()

    obs = env.reset()
    env.render()

    agent = RandomAgent(env.action_space)

    reward = 0
    done = False
    
    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)
        print('action=', action, 'obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break

    env.close()
