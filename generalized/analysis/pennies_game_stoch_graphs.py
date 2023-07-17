import pandas as pd
import matplotlib.pyplot as plt

algaater_compressed = pd.read_csv('./pennies_game_stoch/algaater_compressed.csv')
algaater_compressed['Agent'] = ['Algaater'] * len(algaater_compressed)
algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

bbl_compressed = pd.read_csv('./pennies_game_stoch/bbl_compressed.csv')
bbl_compressed['Agent'] = ['BBL'] * len(bbl_compressed)
bbl_compressed.rename(columns={'BBL': 'Rewards'}, inplace=True)

eee_compressed = pd.read_csv('./pennies_game_stoch/eee_compressed.csv')
eee_compressed['Agent'] = ['EEE'] * len(eee_compressed)
eee_compressed.rename(columns={'EEE': 'Rewards'}, inplace=True)

spp_compressed = pd.read_csv('./pennies_game_stoch/spp_compressed.csv')
spp_compressed['Agent'] = ['S++'] * len(spp_compressed)
spp_compressed.rename(columns={'S++': 'Rewards'}, inplace=True)

combined_df = pd.concat([algaater_compressed, bbl_compressed, eee_compressed, spp_compressed], axis=0,
                        ignore_index=True)
combined_df.to_csv('./pennies_game_stoch/combined_compressed.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed['Rewards'], bbl_compressed['Rewards'], eee_compressed['Rewards'],
             spp_compressed['Rewards']], labels=['AlgAATer', 'BBL', 'EEE', 'S++'])
plt.xlabel('Agent')
plt.ylabel('Rewards')
plt.title('Agent Rewards - Matching Pennies')
plt.savefig(f'./pennies_game_stoch/agent_rewards.png', bbox_inches='tight')

