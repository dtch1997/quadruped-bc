import unittest
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from quadruped_bc.net.agent import Agent

from unittest.mock import MagicMock

def layer_init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0)
    return layer

class AgentTestCase(unittest.TestCase):

    def setUp(self):
        dummy_env = MagicMock()
        dummy_env.single_observation_space.shape = (35,)
        dummy_env.single_action_space.shape = (12,)

        self.envs = dummy_env  # Provide the necessary environment object
        self.agent = Agent(self.envs)

    def test_critic_returns_tensor(self):
        x = torch.tensor(np.random.rand(1, np.array(self.envs.single_observation_space.shape).prod()), dtype=torch.float32)
        value = self.agent.get_value(x)

        self.assertIsInstance(value, torch.Tensor)

    def test_actor_returns_tensor(self):
        x = torch.tensor(np.random.rand(1, np.array(self.envs.single_observation_space.shape).prod()), dtype=torch.float32)
        action, log_prob, entropy, critic_value = self.agent.get_action_and_value(x)

        self.assertIsInstance(action, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertIsInstance(entropy, torch.Tensor)
        self.assertIsInstance(critic_value, torch.Tensor)

    def test_actor_shape_matches_action_space(self):
        x = torch.tensor(np.random.rand(1, np.array(self.envs.single_observation_space.shape).prod()), dtype=torch.float32)
        action, _, _, _ = self.agent.get_action_and_value(x)

        expected_shape = self.envs.single_action_space.shape
        self.assertEqual(action.shape, torch.Size([1] + list(expected_shape)))

if __name__ == '__main__':
    unittest.main()
