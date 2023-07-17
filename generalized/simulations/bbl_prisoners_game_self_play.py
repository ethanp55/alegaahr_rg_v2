from generalized.games.prisoner_game import PrisonersGameMDP
from generalized.agents.algaater import Algaater
from generalized.agents.bbl import BBL
from generalized.games.general_game_items import P1, P2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

prisoners_game = PrisonersGameMDP()

n_epochs = 500
min_rounds = 50
max_rounds = 100
possible_rounds = list(range(min_rounds, max_rounds + 1))
total_rewards_1 = []
total_rewards_2 = []

for epoch in range(1, n_epochs + 1):
    print('Epoch: ' + str(epoch))

    epoch_rewards_1 = []
    epoch_rewards_2 = []

    bbl_idx_1 = np.random.choice([P1, P2])
    bbl_idx_2 = 1 - bbl_idx_1

    expert_agents_1 = Algaater.create_aat_experts(prisoners_game, bbl_idx_1)
    expert_agents_2 = Algaater.create_aat_experts(prisoners_game, bbl_idx_2)

    bbl_1 = BBL('BBL1', prisoners_game, bbl_idx_1)
    bbl_2 = BBL('BBL2', prisoners_game, bbl_idx_2)

    # n_rounds = np.random.choice(possible_rounds)
    n_rounds = min_rounds

    reward_map = {bbl_1.name: 0, bbl_2.name: 0}

    prev_reward_1 = 0
    prev_reward_2 = 0

    for round_num in range(n_rounds):
        print('Round: ' + str(round_num + 1))
        prisoners_game.reset()
        state = deepcopy(prisoners_game.get_init_state())
        action_map = dict()

        key_agent_map = {bbl_1.name: bbl_1, bbl_2.name: bbl_2} if bbl_idx_1 == P1 else \
            {bbl_2.name: bbl_2, bbl_1.name: bbl_1}

        rewards_1 = []
        rewards_2 = []
        actions_p1 = []
        actions_p2 = []
        full_actions1 = []
        full_actions2 = []

        while not state.is_terminal():
            for agent_key, agent in key_agent_map.items():
                agent_reward = prev_reward_1 if agent_key == bbl_1.name else prev_reward_2
                agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                if agent.player == P1:
                    full_actions1.append(agent_action1)

                else:
                    full_actions2.append(agent_action2)

            if state.turn is None or state.turn == bbl_1.player:
                actions_p1.append(action_map[bbl_1.name])

            if state.turn is None or state.turn == bbl_2.player:
                actions_p2.append(action_map[bbl_2.name])

            updated_rewards_map, next_state = prisoners_game.execute_agent_action(action_map)

            for agent_name, new_reward in updated_rewards_map.items():
                reward_map[agent_name] += new_reward

                if agent_name == bbl_1.name:
                    rewards_1.append(new_reward)

                else:
                    rewards_2.append(new_reward)

            state = next_state

        prev_reward_1 = sum(rewards_1)
        prev_reward_2 = sum(rewards_2)
        epoch_rewards_1.append(reward_map[bbl_1.name])
        epoch_rewards_2.append(reward_map[bbl_2.name])

        bbl_1.record_opp_actions(round_num, actions_p2)
        bbl_2.record_opp_actions(round_num, actions_p1)

    total_rewards_1.append(epoch_rewards_1)
    total_rewards_2.append(epoch_rewards_2)

vals = []

for i in range(len(total_rewards_1)):
    final_reward_1, final_reward_2 = total_rewards_1[i][-1], total_rewards_2[i][-1]

    vals.extend([final_reward_1, final_reward_2])

compressed_rewards_df = pd.DataFrame(vals, columns=['BBL'])
compressed_rewards_df.to_csv(f'../analysis/{str(prisoners_game)}/bbl_self_play.csv')

test_results = np.array(total_rewards_1).reshape(n_epochs, -1)
opponent_test_results = np.array(total_rewards_2).reshape(n_epochs, -1)

mean_test_results = test_results.mean(axis=0)
mean_opponent_results = opponent_test_results.mean(axis=0)

x_vals = list(range(test_results.shape[1]))

plt.plot(x_vals, mean_test_results, label='BBL')
plt.plot(x_vals, mean_opponent_results, color='red', label='BBL Mirror')
plt.title('Self Play Rewards')
plt.xlabel('Round #')
plt.ylabel('Rewards ($)')
plt.legend(loc="upper left")
plt.savefig(f'../simulations/{str(prisoners_game)}/bbl_self_play.png', bbox_inches='tight')
plt.clf()

print(f'BBL1: {mean_test_results[-1]}')
print(f'BBL2: {mean_opponent_results[-1]}')













