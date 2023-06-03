import gymnasium as gym 
import quadruped_bc.envs

def test_my_ant_env():
    env = gym.make("MyAnt-v4")

def test_quadruped_env():
    env = gym.make("A1-v4")