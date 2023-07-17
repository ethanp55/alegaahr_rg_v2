from simple_rl.agents.AgentClass import Agent
from generalized.agents.folk_egal import FolkEgalAgent
from generalized.games.microgrid_game import MicrogridGameMDP
import random
import numpy as np


class Random(Agent):
    def __init__(self, name: str, player: int):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.player = player

    def act(self, state, reward, round_num):
        p1_available_actions, p2_available_actions = state.get_available_actions()

        return random.choice(p1_available_actions), random.choice(p2_available_actions)


class GreedyUntilNegative(Agent):
    def __init__(self, name: str, player: int):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.player = player

        microgrid_game = MicrogridGameMDP()
        initial_state = microgrid_game.get_init_state()
        game_name = str(microgrid_game)

        self.bully_agent = FolkEgalAgent('GreedyNegBully', 1, 1, initial_state, game_name + '_bully',
                                         read_from_file=True, specific_policy=True, p1_weight=1.0, player=player)
        self.coop_agent = FolkEgalAgent('GreedyNegCoop', 1, 1, initial_state, game_name, read_from_file=True,
                                        player=player)

    def act(self, state, reward, round_num):
        if reward < 0:
            return self.coop_agent.act(state, reward, round_num)

        else:
            return self.bully_agent.act(state, reward, round_num)


class CoopOrGreedy(Agent):
    def __init__(self, name: str, player: int, defect_proba=0.25):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.player = player

        microgrid_game = MicrogridGameMDP()
        initial_state = microgrid_game.get_init_state()
        game_name = str(microgrid_game)

        self.bully_agent = FolkEgalAgent('CoopGreedyBully', 1, 1, initial_state, game_name + '_bully',
                                         read_from_file=True, specific_policy=True, p1_weight=1.0, player=player)
        self.coop_agent = FolkEgalAgent('CoopGreedyCoop', 1, 1, initial_state, game_name, read_from_file=True,
                                        player=player)
        self.defect_proba = defect_proba

    def act(self, state, reward, round_num):
        defect = np.random.choice([1, 0], p=[self.defect_proba, 1 - self.defect_proba])

        if defect:
            return self.bully_agent.act(state, reward, round_num)

        else:
            return self.coop_agent.act(state, reward, round_num)


class RoundNum(Agent):
    def __init__(self, name: str, player: int):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.player = player

        microgrid_game = MicrogridGameMDP()
        initial_state = microgrid_game.get_init_state()
        game_name = str(microgrid_game)

        self.bully_agent = FolkEgalAgent('PlayNumBully', 1, 1, initial_state, game_name + '_bully',
                                         read_from_file=True, specific_policy=True, p1_weight=1.0, player=player)
        self.coop_agent = FolkEgalAgent('PlayNumCoop', 1, 1, initial_state, game_name, read_from_file=True,
                                        player=player)

    def act(self, state, reward, round_num):
        if round_num >= 24:
            return self.bully_agent.act(state, reward, round_num)

        else:
            return self.coop_agent.act(state, reward, round_num)