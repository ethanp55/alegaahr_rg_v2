import pandas as pd
import matplotlib.pyplot as plt

algaater_compressed = pd.read_csv('./coordination_game/algaater_compressed.csv')
algaater_compressed['Agent'] = ['Algaater'] * len(algaater_compressed)
algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

bbl_compressed = pd.read_csv('./coordination_game/bbl_compressed.csv')
bbl_compressed['Agent'] = ['BBL'] * len(bbl_compressed)
bbl_compressed.rename(columns={'BBL': 'Rewards'}, inplace=True)

eee_compressed = pd.read_csv('./coordination_game/eee_compressed.csv')
eee_compressed['Agent'] = ['EEE'] * len(eee_compressed)
eee_compressed.rename(columns={'EEE': 'Rewards'}, inplace=True)

spp_compressed = pd.read_csv('./coordination_game/spp_compressed.csv')
spp_compressed['Agent'] = ['S++'] * len(spp_compressed)
spp_compressed.rename(columns={'S++': 'Rewards'}, inplace=True)

combined_df = pd.concat([algaater_compressed, bbl_compressed, eee_compressed, spp_compressed], axis=0,
                        ignore_index=True)
combined_df.to_csv('./coordination_game/combined_compressed.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed['Rewards'], bbl_compressed['Rewards'], eee_compressed['Rewards'],
             spp_compressed['Rewards']], labels=['AlgAATer', 'BBL', 'EEE', 'S++'])
plt.xlabel('Agent')
plt.ylabel('Rewards')
plt.title('Agent Rewards - Coordination Game')
plt.savefig(f'./coordination_game/agent_rewards.png', bbox_inches='tight')
plt.clf()

algaater_compressed_test = pd.read_csv('./coordination_game/algaater_compressed_test.csv')
algaater_compressed_test['Agent'] = ['Algaater'] * len(algaater_compressed_test)
algaater_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

bbl_compressed_test = pd.read_csv('./coordination_game/bbl_compressed_test.csv')
bbl_compressed_test['Agent'] = ['BBL'] * len(bbl_compressed_test)
bbl_compressed_test.rename(columns={'BBL': 'Rewards'}, inplace=True)

eee_compressed_test = pd.read_csv('./coordination_game/eee_compressed_test.csv')
eee_compressed_test['Agent'] = ['EEE'] * len(eee_compressed_test)
eee_compressed_test.rename(columns={'EEE': 'Rewards'}, inplace=True)

spp_compressed_test = pd.read_csv('./coordination_game/spp_compressed_test.csv')
spp_compressed_test['Agent'] = ['S++'] * len(spp_compressed_test)
spp_compressed_test.rename(columns={'S++': 'Rewards'}, inplace=True)

combined_df_test = pd.concat([algaater_compressed_test, bbl_compressed_test, eee_compressed_test, spp_compressed_test],
                             axis=0, ignore_index=True)
combined_df_test.to_csv('./coordination_game/combined_compressed_test.csv')
combined_df_test.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed_test['Rewards'], bbl_compressed_test['Rewards'], eee_compressed_test['Rewards'],
             spp_compressed_test['Rewards']], labels=['AlgAATer', 'BBL', 'EEE', 'S++'])
plt.xlabel('Agent')
plt.ylabel('Rewards')
plt.title('Agent Rewards - Coordination Game')
plt.savefig(f'./coordination_game/agent_rewards_test.png', bbox_inches='tight')
plt.clf()

algaater_compressed_test = pd.read_csv('./coordination_game/algaater_compressed_test_changers.csv')
algaater_compressed_test['Agent'] = ['Algaater'] * len(algaater_compressed_test)
algaater_compressed_test.rename(columns={'Algaater': 'Rewards'}, inplace=True)

bbl_compressed_test = pd.read_csv('./coordination_game/bbl_compressed_test_changers.csv')
bbl_compressed_test['Agent'] = ['BBL'] * len(bbl_compressed_test)
bbl_compressed_test.rename(columns={'BBL': 'Rewards'}, inplace=True)

eee_compressed_test = pd.read_csv('./coordination_game/eee_compressed_test_changers.csv')
eee_compressed_test['Agent'] = ['EEE'] * len(eee_compressed_test)
eee_compressed_test.rename(columns={'EEE': 'Rewards'}, inplace=True)

spp_compressed_test = pd.read_csv('./coordination_game/spp_compressed_test_changers.csv')
spp_compressed_test['Agent'] = ['S++'] * len(spp_compressed_test)
spp_compressed_test.rename(columns={'S++': 'Rewards'}, inplace=True)

combined_df_test = pd.concat([algaater_compressed_test, bbl_compressed_test, eee_compressed_test, spp_compressed_test],
                             axis=0, ignore_index=True)
combined_df_test.to_csv('./coordination_game/combined_compressed_test_changers.csv')
combined_df_test.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed_test['Rewards'], bbl_compressed_test['Rewards'], eee_compressed_test['Rewards'],
             spp_compressed_test['Rewards']], labels=['AlgAATer', 'BBL', 'EEE', 'S++'])
plt.xlabel('Agent')
plt.ylabel('Rewards')
plt.title('Agent Rewards - Coordination Game')
plt.savefig(f'./coordination_game/agent_rewards_test_changers.png', bbox_inches='tight')
plt.clf()

