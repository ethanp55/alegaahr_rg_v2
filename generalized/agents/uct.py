from generalized.games.GameState import GameState
from simple_rl.agents.AgentClass import Agent
from generalized.games.general_game_items import P1, P2
import numpy as np
import random


class UCTGameNode:
    def __init__(self, game_state: GameState, action1=None, action2=None, reward1: float = 0, reward2: float = 0,
                 n_visits: int = 0):
        self.game_state = game_state
        self.action1 = action1
        self.action2 = action2
        self.reward1 = reward1
        self.reward2 = reward2
        self.n_visits = n_visits


class UCTAgent(Agent):
    def __init__(self, name: str, n_iterations: int, decision_func, decision_func_simultaneous) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.n_iterations = n_iterations
        self.decision_func = decision_func
        self.decision_func_simultaneous = decision_func_simultaneous

    def act(self, state: GameState, reward: float):
        root = UCTGameNode(state)
        simultaneous = state.is_simultaneous()

        def random_walk(curr_state: GameState):
            if curr_state.is_terminal():
                return curr_state.reward(P1), curr_state.reward(P2)

            p1_available_actions, p2_available_actions = curr_state.get_available_actions()

            p1_action = random.choice(p1_available_actions)
            p2_action = random.choice(p2_available_actions)

            next_state = curr_state.next(p1_action, p2_action)

            return random_walk(next_state)

        if simultaneous:
            node_map1 = {str(state): root}
            node_map2 = {str(state): root}

            def rec_search_simultaneous(curr_node: UCTGameNode):
                curr_state = curr_node.game_state

                if curr_state.is_terminal():
                    return curr_state.reward(P1), curr_state.reward(P2)

                elif curr_node.n_visits == 0:
                    curr_node.n_visits += 1
                    return random_walk(curr_state)

                curr_node.n_visits += 1

                p1_available_actions, p2_available_actions = curr_state.get_available_actions()

                if simultaneous:
                    children1 = {}
                    children2 = {}

                    for action in p1_available_actions:
                        for opp_action in p2_available_actions:
                            next_state = curr_state.next(action, opp_action)
                            child_node = node_map1.get(str(next_state), None)

                            if child_node is None:
                                child_node = UCTGameNode(next_state, action, opp_action)
                                node_map1[str(next_state)] = child_node

                            children1[action] = children1.get(action, []) + [child_node]

                    for action in p2_available_actions:
                        for opp_action in p1_available_actions:
                            next_state = curr_state.next(opp_action, action)
                            child_node = node_map2.get(str(next_state), None)

                            if child_node is None:
                                child_node = UCTGameNode(next_state, opp_action, action)
                                node_map2[str(next_state)] = child_node

                            children2[action] = children2.get(action, []) + [child_node]

                    node_to_explore1 = self.decision_func_simultaneous(children1, P1, curr_node.n_visits)

                    reward1, _ = rec_search_simultaneous(node_to_explore1)
                    curr_node.reward1 += reward1

                    node_to_explore2 = self.decision_func_simultaneous(children2, P2, curr_node.n_visits)

                    _, reward2 = rec_search_simultaneous(node_to_explore2)
                    curr_node.reward2 += reward2

                    return reward1, reward2

            for n in range(self.n_iterations):
                rec_search_simultaneous(root)

            p1_actions, p2_actions = state.get_available_actions()
            c1 = {}
            c2 = {}

            for a in p1_actions:
                opp_a = random.choice(p2_actions)
                next_s = state.next(a, opp_a)
                c_node = node_map1.get(str(next_s), None)

                if c_node is not None:
                    c1[a] = c1.get(a, []) + [c_node]

            chosen_node1 = self.decision_func_simultaneous(c1, P1, root.n_visits)

            for a in p2_actions:
                opp_a = random.choice(p1_actions)
                next_s = state.next(opp_a, a)
                c_node = node_map2.get(str(next_s), None)

                if c_node is not None:
                    c2[a] = c2.get(a, []) + [c_node]

            chosen_node2 = self.decision_func_simultaneous(c2, P2, root.n_visits)

            return chosen_node1.action1, chosen_node2.action2

        else:
            node_map = {str(state): root}

            def rec_search(curr_node: UCTGameNode):
                curr_state = curr_node.game_state

                if curr_state.is_terminal():
                    return curr_state.reward(P1), curr_state.reward(P2)

                elif curr_node.n_visits == 0:
                    curr_node.n_visits += 1
                    return random_walk(curr_state)

                curr_node.n_visits += 1

                p1_available_actions, p2_available_actions = curr_state.get_available_actions()

                children = []
                curr_turn = curr_state.turn

                available_actions = p1_available_actions if curr_turn == P1 else p2_available_actions
                opp_available_actions = p2_available_actions if curr_turn == P1 else p1_available_actions

                for action in available_actions:
                    opp_action = random.choice(opp_available_actions)
                    next_state = curr_state.next(action, opp_action) if curr_turn == P1 else \
                        curr_state.next(opp_action, action)
                    child_node = node_map.get(str(next_state), None)

                    if child_node is None:
                        child_node = UCTGameNode(next_state, action, opp_action) if curr_turn == P1 else \
                            UCTGameNode(next_state, opp_action, action)
                        node_map[str(next_state)] = child_node

                    children.append(child_node)

                node_to_explore = self.decision_func(children, curr_turn, curr_node.n_visits)

                reward1, reward2 = rec_search(node_to_explore)
                curr_node.reward1 += reward1
                curr_node.reward2 += reward2

                return reward1, reward2

            for n in range(self.n_iterations):
                rec_search(root)

            p1_actions, p2_actions = state.get_available_actions()
            c = []
            turn = state.turn

            actions = p1_actions if turn == P1 else p2_actions
            opp_actions = p2_actions if turn == P1 else p1_actions

            for a in actions:
                opp_a = random.choice(opp_actions)
                next_s = state.next(a, opp_a) if turn == P1 else state.next(opp_a, a)
                c_node = node_map.get(str(next_s), None)

                if c_node is not None:
                    c.append(c_node)

            chosen_node = self.decision_func(c, turn, root.n_visits, True)

            return chosen_node.action1, chosen_node.action2


class MaxSelfAgent(UCTAgent):
    def __init__(self, name: str, n_iterations: int):
        self.name = name
        self.n_iterations = n_iterations

        def decision_func(children, player, curr_visits, is_root=False):
            max_val = -np.inf
            max_node = None

            for node in children:
                if node.n_visits == 0:
                    return node

                node_reward = (node.reward1 if player == P1 else node.reward2) / node.n_visits

                node_val = node_reward if is_root else node_reward + 2 * ((np.log(curr_visits) / node.n_visits) ** 0.5)

                if node_val > max_val:
                    max_val = node_val
                    max_node = node

            return max_node

        def decision_func_simultaneous(children, player, curr_visits):
            max_val = -np.inf
            max_action = None

            for action, nodes in children.items():
                action_avg = 0

                for node in nodes:
                    if node.n_visits == 0:
                        return node

                    node_reward = (node.reward1 if player == P1 else node.reward2) / node.n_visits
                    action_avg += node_reward + 2 * ((np.log(curr_visits) / node.n_visits) ** 0.5)

                action_avg /= len(nodes)

                if action_avg > max_val:
                    max_val = action_avg
                    max_action = action

            return random.choice(children[max_action])

        UCTAgent.__init__(self, self.name, self.n_iterations, decision_func, decision_func_simultaneous)


class BulliedAgent(UCTAgent):
    def __init__(self, name: str, n_iterations: int):
        self.name = name
        self.n_iterations = n_iterations

        def decision_func(children, player, curr_visits, is_root=False):
            min_val = np.inf
            min_node = None

            for node in children:
                if node.n_visits == 0:
                    return node

                node_reward = (node.reward1 if player == P1 else node.reward2) / node.n_visits
                node_val = node_reward if is_root else node_reward + 2 * ((np.log(curr_visits) * node.n_visits) ** 0.5)

                if 0 < node_val < min_val:
                    min_val = node_val
                    min_node = node

            if min_node is None:
                return random.choice(children)

            return min_node

        def decision_func_simultaneous(children, player, curr_visits):
            min_val = np.inf
            min_action = None

            for action, nodes in children.items():
                action_avg = 0

                for node in nodes:
                    if node.n_visits == 0:
                        return node

                    node_reward = (node.reward1 if player == P1 else node.reward2) / node.n_visits
                    action_avg += node_reward + 2 * ((np.log(curr_visits) / node.n_visits) ** 0.5)

                action_avg /= len(nodes)

                if 0 < action_avg < min_val:
                    min_val = action_avg
                    min_action = action

            if min_action is None:
                min_action = random.choice(list(children.keys()))

            return random.choice(children[min_action])

        UCTAgent.__init__(self, self.name, self.n_iterations, decision_func, decision_func_simultaneous)
