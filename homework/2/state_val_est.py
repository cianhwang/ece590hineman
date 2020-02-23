from gridworld_env import *
from random_agent import *
import numpy as np
from vpg_agent import *

def estStateVal(lr = 0.05, max_iter = 20000):

    env = env_fn()
    s_table = np.random.randn(5, 5)
    n_steps = 20
    reward = 0
    agent = RandomAgent(env.action_space)
#    agent = VpgAgent()
    t = 0
    res = 100.0
    while res > 0.0001 and t < max_iter:
        reward = 0
        done = False
        obs = env.reset()        
        s_table_temp = s_table.copy()
        for step in range(n_steps):
            x_p, y_p = obs.astype(np.int32)
            action = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(action)
            x, y = obs.astype(np.int32)
            s_table[x_p, y_p] += lr * (0.9*s_table[x, y] + reward - s_table[x_p, y_p])
            if done:
                break
        res = np.sum(np.abs(s_table - s_table_temp))

        t += 1
        if t % 1000 == 1000-1:
            print("round {}, res = {:.5f}, table = ".format(t, res))
            print(np.round(s_table,1))
    print(np.round(s_table,1))

if __name__ == '__main__':

    estStateVal()
