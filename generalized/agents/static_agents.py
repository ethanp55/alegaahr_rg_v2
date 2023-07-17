from generalized.games.GameState import GameState
from simple_rl.agents.AgentClass import Agent
from generalized.games.general_game_items import P1, P2
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np

AVAILABLE = 3


class GameTreeNode:
    def __init__(self, state_id: int, state: GameState, action_to_children_map: Dict[Tuple[str, str], 'GameTreeNode'],
                 visited: bool = False) -> None:
        self.state_id = state_id
        self.state = state
        self.action_to_children_map = action_to_children_map
        self.visited = visited


class GameTree:
    def __init__(self, initial_state: GameState) -> None:
        # Generate the tree by recursively creating children, starting at the very first state of the game
        self.simultaneous = initial_state.is_simultaneous()
        self.head_node: GameTreeNode = self._create_children(initial_state)

        self.state_id_map: Dict[int, GameTreeNode] = {}

        # Use breadth-first search to label each state id in the tree (top of the tree is smallest, bottom is largest)
        self._bfs(self.head_node)

    def _create_children(self, parent_state: GameState) -> GameTreeNode:
        if parent_state.is_terminal():
            return GameTreeNode(0, parent_state, {})

        p1_available_actions, p2_available_actions = parent_state.get_available_actions()

        if self.simultaneous:
            simultaneous_action_children_map = {}

            for action1 in p1_available_actions:
                for action2 in p2_available_actions:
                    new_state = parent_state.next(action1, action2)
                    child_node = self._create_children(new_state)
                    simultaneous_action_children_map[(action1, action2)] = child_node

            return GameTreeNode(0, parent_state, simultaneous_action_children_map)

        else:
            curr_turn = parent_state.turn

            action_children_map = {}
            available_actions = p1_available_actions if curr_turn == P1 else p2_available_actions

            for action in available_actions:
                new_state = parent_state.next(action, action)
                child_node = self._create_children(new_state)
                action_children_map[(action, action)] = child_node

            return GameTreeNode(0, parent_state, action_to_children_map=action_children_map)

    def _bfs(self, head_node: GameTreeNode) -> None:
        state_id = 0
        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(head_node)
        head_node.visited = True
        head_node.state_id = state_id
        self.state_id_map[state_id] = head_node

        while queue:
            # Dequeue a node from the queue
            new_node = queue.pop(0)

            # Get all children nodes of the dequeued node new_node. If a child node has not been visited, mark and
            # enqueue it
            for child_node in new_node.action_to_children_map.values():
                if not child_node.visited:
                    state_id += 1
                    queue.append(child_node)
                    child_node.visited = True
                    child_node.state_id = state_id
                    self.state_id_map[state_id] = child_node


class StaticGameAgent(Agent):
    def __init__(self, eval_func: 'function', name: str, game_tree: GameTree) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.eval_func = eval_func
        self.name = name
        self.state_to_action_map: Dict[str, Tuple[str, str]] = {}
        self._train(game_tree)

        def policy(state: GameState, reward) -> Tuple[str, str]:
            return self.state_to_action_map[str(state)]

        self.policy = policy

    def _train(self, game_tree: GameTree):
        state_map = OrderedDict(sorted(game_tree.state_id_map.items(), reverse=True))
        ideal_reward_map: Dict[GameTreeNode, Tuple[float, float]] = {}

        for tree_node in state_map.values():
            if tree_node.state.is_terminal():
                p1_reward = tree_node.state.reward(P1)
                p2_reward = tree_node.state.reward(P2)

                ideal_reward_map[tree_node] = p1_reward, p2_reward
                self.state_to_action_map[str(tree_node.state)] = None, None

            else:
                reward_action_pairs = [(ideal_reward_map[tree_node.action_to_children_map[action_tup]], action_tup) for
                                       action_tup in tree_node.action_to_children_map.keys()]

                ideal_reward1, ideal_reward2, ideal_action1, ideal_action2 = self.eval_func(reward_action_pairs, tree_node.state)

                ideal_reward_map[tree_node] = ideal_reward1, ideal_reward2
                self.state_to_action_map[str(tree_node.state)] = ideal_action1, ideal_action2

    def act(self, state: GameState, reward):
        return self.policy(state, reward)

    def __str__(self) -> str:
        return str(self.name)


def max_self_func(reward_action_pairs: List[Tuple[Tuple[float, float], Tuple[str, str]]], state: GameState) -> Tuple[float, float, str, str]:
    if state.is_simultaneous():
        max_reward1, max_reward2 = -np.inf, -np.inf
        max_action1, max_action2 = None, None

        for reward1, reward2, action1, action2 in reward_action_pairs:
            if reward1 > max_reward1:
                max_reward1 = reward1
                max_action1 = action1

            if reward2 > max_reward2:
                max_reward2 = reward2
                max_action2 = action2

        return max_reward1, max_reward2, max_action1, max_action2

    curr_turn = state.turn
    max_item = max(reward_action_pairs, key=lambda x: x[0][curr_turn])

    return max_item[0][0], max_item[0][1], max_item[1][0], max_item[1][1]


def bullied_func(reward_action_pairs: List[Tuple[Tuple[float, float], Tuple[str, str]]], state: GameState) -> Tuple[float, float, str, str]:
    if state.is_simultaneous():
        min_reward1, min_reward2 = np.inf, np.inf
        min_action1, min_action2 = None, None

        max_reward1, max_reward2 = -np.inf, -np.inf
        max_action1, max_action2 = None, None

        for reward1, reward2, action1, action2 in reward_action_pairs:
            if 0 <= reward1 < min_reward1:
                min_reward1 = reward1
                min_action1 = action1

            if 0 <= reward2 < min_reward2:
                min_reward2 = reward2
                min_action2 = action2

            if reward1 > max_reward1:
                max_reward1 = reward1
                max_action1 = action1

            if reward2 > max_reward2:
                max_reward2 = reward2
                max_action2 = action2

        reward1 = min_reward1 if min_action1 is not None else max_reward1
        reward2 = min_reward2 if min_action2 is not None else max_reward2
        action1 = min_action1 if min_action1 is not None else max_action1
        action2 = min_action2 if min_action2 is not None else max_action2

        return reward1, reward2, action1, action2

    curr_turn = state.turn
    possible_entries = [tup for tup in reward_action_pairs if tup[0][curr_turn] >= 0]

    if len(possible_entries) == 0:
        max_item = max(reward_action_pairs, key=lambda x: x[0][curr_turn])

        return max_item[0][0], max_item[0][1], max_item[1][0], max_item[1][1]

    min_item = min(possible_entries, key=lambda x: x[0][curr_turn])

    return min_item[0][0], min_item[0][1], min_item[1][0], min_item[1][1]



