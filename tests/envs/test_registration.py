import gymnasium as gym 
import quadruped_bc.envs

def test_my_ant_env():
    env = gym.make("MyAnt-v4")