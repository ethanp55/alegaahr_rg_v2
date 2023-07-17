import pandas as pd
import matplotlib.pyplot as plt

algaater_compressed = pd.read_csv('./pennies_game/algaater_self_play.csv')
algaater_compressed['Agent'] = ['Algaater'] * len(algaater_compressed)
algaater_compressed.rename(columns={'Algaater': 'Rewards'}, inplace=True)

bbl_compressed = pd.read_csv('./pennies_game/bbl_self_play.csv')
bbl_compressed['Agent'] = ['BBL'] * len(bbl_compressed)
bbl_compressed.rename(columns={'BBL': 'Rewards'}, inplace=True)

eee_compressed = pd.read_csv('./pennies_game/eee_self_play.csv')
eee_compressed['Agent'] = ['EEE'] * len(eee_compressed)
eee_compressed.rename(columns={'EEE': 'Rewards'}, inplace=True)

spp_compressed = pd.read_csv('./pennies_game/spp_self_play.csv')
spp_compressed['Agent'] = ['S++'] * len(spp_compressed)
spp_compressed.rename(columns={'S++': 'Rewards'}, inplace=True)

combined_df = pd.concat([algaater_compressed, bbl_compressed, eee_compressed, spp_compressed], axis=0,
                        ignore_index=True)
combined_df.to_csv('./pennies_game/combined_self_play.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed['Rewards'], bbl_compressed['Rewards'], eee_compressed['Rewards'],
             spp_compressed['Rewards']], labels=['AlgAATer', 'BBL', 'EEE', 'S++'])
plt.xlabel('Agent')
plt.ylabel('Rewards')
plt.title('Self Play Rewards - Matching Pennies')
plt.savefig(f'./pennies_game/agent_rewards_self_play.png', bbox_inches='tight')
plt.clf()
