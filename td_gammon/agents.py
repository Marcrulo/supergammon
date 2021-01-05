import random
import time
from itertools import count
from random import randint, choice

import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS

random.seed(0)


# AGENT ============================================================================================


class Agent:
    def __init__(self, color):
        self.color = color
        self.name = 'Agent({})'.format(COLORS[color])

    def roll_dice(self):
        return (-randint(1, 6), -randint(1, 6)) if self.color == WHITE else (randint(1, 6), randint(1, 6))

    def choose_best_action(self, actions, env):
        raise NotImplementedError


# RANDOM AGENT =======================================================================================


class RandomAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'RandomAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        return choice(list(actions)) if actions else None


# HUMAN AGENT =======================================================================================


class HumanAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'HumanAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions=None, env=None):
        pass


# TD-GAMMON AGENT =====================================================================================


class TDAgent(Agent):
    def __init__(self, color, net):
        super().__init__(color)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation)

                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            # practical-issues-in-temporal-difference-learning, pag.3
            # ... the network's output P_t is an estimate of White's probability of winning from board position x_t.
            # ... the move which is selected at each time step is the move which maximizes P_t when White is to play and minimizes P_t when Black is to play.
            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action

