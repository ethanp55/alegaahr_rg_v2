from generalized.games.chicken_game import ChickenGameMDP
from generalized.agents.algaater import Algaater
from generalized.agents.bbl import BBL
from generalized.agents.eee import EEE
from generalized.games.general_game_items import P1, P2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

chicken_game = ChickenGameMDP()

data_dir = '../training/training_data/' + str(chicken_game) + '/'

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

    bbl_idx = np.random.choice([P1, P2])
    eee_idx = 1 - bbl_idx
    eee_experts = Algaater.create_aat_experts(chicken_game, eee_idx)

    print('BBL: ' + str(bbl_idx))
    print('EEE: ' + str(eee_idx))

    bbl = BBL('BBL', chicken_game, bbl_idx)
    eee = EEE('EEE', eee_experts, eee_idx, demo=True)

    # n_rounds = np.random.choice(possible_rounds)
    n_rounds = min_rounds

    reward_map = {bbl.name: 0, eee.name: 0}

    prev_reward_1 = 0
    prev_reward_2 = 0

    for round_num in range(n_rounds):
        print('Round: ' + str(round_num + 1))
        chicken_game.reset()
        state = deepcopy(chicken_game.get_init_state())
        action_map = dict()
        opp_actions_1 = []
        opp_actions_2 = []

        key_agent_map = {bbl.name: bbl, eee.name: eee} if bbl_idx == P1 else \
            {eee.name: eee, bbl.name: bbl}

        rewards_1 = []
        rewards_2 = []

        while not state.is_terminal():
            for agent_key, agent in key_agent_map.items():
                agent_reward = prev_reward_1 if agent_key == bbl.name else prev_reward_2
                agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                if agent_key == eee.name:
                    # algaater_1.assumption_checker.act(state, agent_reward, round_num)

                    if state.turn is None or state.turn == eee_idx:
                        opp_action = agent_action1 if eee.player == P1 else agent_action2
                        opp_actions_2.append(opp_action)

                else:
                    # algaater_2.assumption_checker.act(state, agent_reward, round_num)

                    if state.turn is None or state.turn == bbl_idx:
                        opp_action = agent_action1 if bbl.player == P1 else agent_action2
                        opp_actions_1.append(opp_action)

            updated_rewards_map, next_state = chicken_game.execute_agent_action(action_map)

            for agent_name, new_reward in updated_rewards_map.items():
                reward_map[agent_name] += new_reward

                if agent_name == bbl.name:
                    rewards_1.append(new_reward)

                else:
                    rewards_2.append(new_reward)

            state = next_state

        prev_reward_1 = sum(rewards_1)
        prev_reward_2 = sum(rewards_2)
        epoch_rewards_1.append(reward_map[eee.name])
        epoch_rewards_2.append(reward_map[bbl.name])
        bbl.record_opp_actions(round_num, opp_actions_2)

    total_rewards_1.append(epoch_rewards_1)
    total_rewards_2.append(epoch_rewards_2)

vals_1, vals_2 = [], []

for i in range(len(total_rewards_1)):
    final_reward_1, final_reward_2 = total_rewards_1[i][-1], total_rewards_2[i][-1]

    vals_1.append(final_reward_1)
    vals_2.append(final_reward_2)

compressed_rewards_df = pd.DataFrame(vals_1, columns=['EEE'])
compressed_rewards_df.to_csv(f'../analysis/{str(chicken_game)}/eee_vs_bbl_eee.csv')

compressed_rewards_df_opp = pd.DataFrame(vals_2, columns=['BBL'])
compressed_rewards_df_opp.to_csv(f'../analysis/{str(chicken_game)}/eee_vs_bbl_bbl.csv')

test_results = np.array(total_rewards_1).reshape(n_epochs, -1)
opponent_test_results = np.array(total_rewards_2).reshape(n_epochs, -1)

mean_test_results = test_results.mean(axis=0)
mean_opponent_results = opponent_test_results.mean(axis=0)

x_vals = list(range(test_results.shape[1]))

plt.plot(x_vals, mean_test_results, label='EEE')
plt.plot(x_vals, mean_opponent_results, color='red', label='BBL')
plt.title('EEE vs. BBL - Chicken Game')
plt.xlabel('Round #')
plt.ylabel('Rewards ($)')
plt.legend(loc="upper left")
plt.savefig(f'../simulations/{str(chicken_game)}/eee_vs_bbl.png', bbox_inches='tight')
plt.clf()

print(f'EEE: {mean_test_results[-1]}')
print(f'BBL: {mean_opponent_results[-1]}')
