from generalized.games.block_game import BlockGameMDP
from generalized.agents.block_game_specific_agents import Random, GreedyUntilNegative, \
    CoopOrGreedy, RoundNum, RoundNum2, RoundNum3
from generalized.games.general_game_items import P1, P2
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from generalized.agents.spp import SPP
from generalized.agents.eee import EEE
from generalized.agents.algaater import Algaater
from generalized.agents.folk_egal import FolkEgalAgent, FolkEgalPunishAgent

block_game = BlockGameMDP()

EXPERT_SET_NAMES = ['CoopOpp', 'CoopPunishOpp', 'BullyOpp', 'BullyPunishOpp', 'BulliedOpp', 'MinimaxOpp', 'CfrOpp']
OTHER_NAMES = ['Random', 'GreedyNeg', 'CoopGreedy', 'RoundNum', 'RoundNum2', 'RoundNum3']
CHANGER_NAMES = ['RoundNum', 'RoundNum2', 'RoundNum3']


# Create opponent agents
def create_opponent_agents(player_idx):
    # Experts in AlgAATer's pool
    initial_state = block_game.get_init_state()
    game_name = str(block_game)

    # Experts in AlgAATer's pool
    coop = FolkEgalAgent('CoopOpp', 1, 1, initial_state, game_name, read_from_file=True, player=player_idx)
    coop_punish = FolkEgalPunishAgent('CoopPunishOpp', coop, game_name, block_game)
    bully = FolkEgalAgent('BullyOpp', 1, 1, initial_state, game_name + '_bully', read_from_file=True,
                          specific_policy=True, p1_weight=1.0, player=player_idx)
    bully_punish = FolkEgalPunishAgent('BullyPunishOpp', bully, game_name, block_game)

    # Other agents
    random_agent = Random('Random', player_idx)
    greedy_neg_agent = GreedyUntilNegative('GreedyNeg', player_idx)
    coop_greedy_agent = CoopOrGreedy('CoopGreedy', player_idx)

    eee_experts = Algaater.create_aat_experts(block_game, player_idx)
    spp = SPP('RoundNum3S++', block_game, player_idx)
    eee = EEE('RoundNum3EEE', eee_experts, player_idx, demo=True)

    round_num_agent = RoundNum('RoundNum', player_idx)
    round_num2_agent = RoundNum2('RoundNum2', player_idx)
    round_num3_agent = RoundNum3('RoundNum3', player_idx, spp, eee)

    opponents = {coop_punish.name: coop_punish, bully.name: bully, coop_greedy_agent.name: coop_greedy_agent,
                 random_agent.name: random_agent, bully_punish.name: bully_punish, greedy_neg_agent.name:
                     greedy_neg_agent, round_num_agent.name: round_num_agent,
                 round_num2_agent.name: round_num2_agent, round_num3_agent.name: round_num3_agent}

    return opponents


create_graphs = True
save_data = True

n_epochs = 500
min_rounds = 50
max_rounds = 100
possible_rounds = list(range(min_rounds, max_rounds + 1))
total_rewards = {}
total_opp_rewards = {}

for epoch in range(1, n_epochs + 1):
    print('Epoch: ' + str(epoch))
    eee_idx = np.random.choice([P1, P2])
    opponent_idx = 1 - eee_idx

    opponents = create_opponent_agents(opponent_idx)
    eee_experts = Algaater.create_aat_experts(block_game, eee_idx)
    eee = EEE('EEE', eee_experts, eee_idx)

    # n_rounds = np.random.choice(possible_rounds)
    n_rounds = min_rounds

    for opponent_key in opponents.keys():
        print('Opponent: ' + str(opponent_key))
        opponent_agent = deepcopy(opponents[opponent_key])
        eee_rewards = []
        opp_rewards = []
        reward_map = {opponent_key: 0, eee.name: 0}
        eee.reset()

        prev_reward_1 = 0
        prev_reward_2 = 0

        for round_num in range(n_rounds):
            print('Round: ' + str(round_num + 1))
            block_game.reset()
            state = deepcopy(block_game.get_init_state())
            action_map = dict()

            key_agent_map = {eee.name: eee, opponent_key: opponent_agent} if eee_idx == P1 else \
                {opponent_key: opponent_agent, eee.name: eee}

            rewards_1 = []
            rewards_2 = []

            while not state.is_terminal():
                for agent_key, agent in key_agent_map.items():
                    agent_reward = prev_reward_1 if agent_key == eee.name else prev_reward_2
                    agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                    action_map[agent_key] = agent_action1 if agent.player == P1 else agent_action2

                updated_rewards_map, next_state = block_game.execute_agent_action(action_map)

                for agent_name, new_reward in updated_rewards_map.items():
                    reward_map[agent_name] += new_reward

                    if agent_name == eee.name:
                        rewards_1.append(new_reward)

                    else:
                        rewards_2.append(new_reward)

                prev_state = deepcopy(state)
                state = next_state

            prev_reward_1 = sum(rewards_1)
            prev_reward_2 = sum(rewards_2)
            agent_reward = reward_map[eee.name]
            eee_rewards.append(agent_reward / 100)
            opp_rewards.append(reward_map[opponent_key] / 100)

        total_rew = total_rewards.get(opponent_key, [])
        total_rew.append(eee_rewards)
        total_rewards[opponent_key] = total_rew
        total_opp_rew = total_opp_rewards.get(opponent_key, [])
        total_opp_rew.append(opp_rewards)
        total_opp_rewards[opponent_key] = total_opp_rew

if save_data:
    vals = []
    vals_test = []
    vals_test_changers = []

    for expert_key, rewards in total_rewards.items():
        for epoch_rewards in rewards:
            if expert_key in EXPERT_SET_NAMES:
                vals.append(epoch_rewards[-1])

            else:
                vals_test.append(epoch_rewards[-1])

            if expert_key in CHANGER_NAMES:
                vals_test_changers.append(epoch_rewards[-1])

    compressed_rewards_df = pd.DataFrame(vals, columns=['EEE'])
    compressed_rewards_test_df = pd.DataFrame(vals_test, columns=['EEE'])
    compressed_rewards_test_changers_df = pd.DataFrame(vals_test_changers, columns=['EEE'])

    columns = []
    columns_test = []
    columns_test_changers = []
    vals = []
    vals_test = []
    vals_test_changers = []

    for agent_name, rewards in total_rewards.items():
        agent_epoch_rewards = []

        for epoch_rewards in rewards:
            agent_epoch_rewards.append(epoch_rewards[-1])

        if agent_name in EXPERT_SET_NAMES:
            vals.append(agent_epoch_rewards)
            columns.append(agent_name)

        else:
            vals_test.append(agent_epoch_rewards)
            columns_test.append(agent_name)

        if agent_name in CHANGER_NAMES:
            vals_test_changers.append(agent_epoch_rewards)
            columns_test_changers.append(agent_name)

    full_rewards_df = pd.DataFrame(zip(*vals), columns=columns)
    full_rewards_test_df = pd.DataFrame(zip(*vals_test), columns=columns_test)
    full_rewards_test_changers_df = pd.DataFrame(zip(*vals_test_changers), columns=columns_test_changers)

    compressed_rewards_df.to_csv(f'../analysis/{str(block_game)}/eee_compressed.csv')
    compressed_rewards_test_df.to_csv(f'../analysis/{str(block_game)}/eee_compressed_test.csv')
    compressed_rewards_test_changers_df.to_csv(f'../analysis/{str(block_game)}/eee_compressed_test_changers.csv')
    full_rewards_df.to_csv(f'../analysis/{str(block_game)}/eee_full.csv')
    full_rewards_test_df.to_csv(f'../analysis/{str(block_game)}/eee_full_test.csv')
    full_rewards_test_changers_df.to_csv(f'../analysis/{str(block_game)}/eee_full_test_changers.csv')

if create_graphs:
    for opponent_key in opponents.keys():
        algaater_epoch_rewards = np.array(total_rewards[opponent_key]).reshape(n_epochs, -1)
        opp_epoch_rewards = np.array(total_opp_rewards[opponent_key]).reshape(n_epochs, -1)

        algaater_mean_rewards = algaater_epoch_rewards.mean(axis=0)
        opp_mean_rewards = opp_epoch_rewards.mean(axis=0)

        x_vals = list(range(algaater_epoch_rewards.shape[1]))

        plt.plot(x_vals, algaater_mean_rewards, label='EEE')
        plt.plot(x_vals, opp_mean_rewards, color='red', label=opponent_key)
        plt.title('EEE vs. ' + str(opponent_key))
        plt.xlabel('Round #')
        plt.ylabel('Rewards ($)')
        plt.legend(loc="upper left")
        plt.savefig(f'../simulations/{str(block_game)}/eee_vs_{opponent_key}.png', bbox_inches='tight')
        plt.clf()
