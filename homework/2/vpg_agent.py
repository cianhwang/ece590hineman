import gym
import spinup
from gridworld_env import *
from spinup.utils.test_policy import load_policy, run_policy

class VpgAgent(object):

    def __init__(self):
        _, self.get_action = load_policy('log2/')

    def act(self, obs, *argv):
        action = self.get_action(obs)
        return action


if __name__ == '__main__':

    spinup.vpg(env_fn)
    '''
    _, get_action = load_policy('log2/')
    env = env_fn()
    obs = env.reset()
    env.render()

    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        action = get_action(obs)
        obs, reward, done, info = env.step(action)
        print('action=', action, 'obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break

    env.close()
'''
