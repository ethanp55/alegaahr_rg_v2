import pandas as pd
import matplotlib.pyplot as plt

algaater_compressed1 = pd.read_csv('./pennies_game/algaater_vs_bbl_algaater.csv')
algaater_compressed1['Agent'] = ['Algaater'] * len(algaater_compressed1)
algaater_compressed1.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_compressed2 = pd.read_csv('./pennies_game/algaater_vs_eee_algaater.csv')
algaater_compressed2['Agent'] = ['Algaater'] * len(algaater_compressed2)
algaater_compressed2.rename(columns={'Algaater': 'Rewards'}, inplace=True)

algaater_compressed3 = pd.read_csv('./pennies_game/algaater_vs_spp_algaater.csv')
algaater_compressed3['Agent'] = ['Algaater'] * len(algaater_compressed3)
algaater_compressed3.rename(columns={'Algaater': 'Rewards'}, inplace=True)

bbl_compressed1 = pd.read_csv('./pennies_game/algaater_vs_bbl_bbl.csv')
bbl_compressed1['Agent'] = ['BBL'] * len(bbl_compressed1)
bbl_compressed1.rename(columns={'BBL': 'Rewards'}, inplace=True)

bbl_compressed2 = pd.read_csv('./pennies_game/eee_vs_bbl_bbl.csv')
bbl_compressed2['Agent'] = ['BBL'] * len(bbl_compressed2)
bbl_compressed2.rename(columns={'BBL': 'Rewards'}, inplace=True)

bbl_compressed3 = pd.read_csv('./pennies_game/bbl_vs_spp_bbl.csv')
bbl_compressed3['Agent'] = ['BBL'] * len(bbl_compressed3)
bbl_compressed3.rename(columns={'BBL': 'Rewards'}, inplace=True)

eee_compressed1 = pd.read_csv('./pennies_game/algaater_vs_eee_eee.csv')
eee_compressed1['Agent'] = ['EEE'] * len(eee_compressed1)
eee_compressed1.rename(columns={'EEE': 'Rewards'}, inplace=True)

eee_compressed2 = pd.read_csv('./pennies_game/eee_vs_bbl_eee.csv')
eee_compressed2['Agent'] = ['EEE'] * len(eee_compressed2)
eee_compressed2.rename(columns={'EEE': 'Rewards'}, inplace=True)

eee_compressed3 = pd.read_csv('./pennies_game/eee_vs_spp_eee.csv')
eee_compressed3['Agent'] = ['EEE'] * len(eee_compressed3)
eee_compressed3.rename(columns={'EEE': 'Rewards'}, inplace=True)

spp_compressed1 = pd.read_csv('./pennies_game/algaater_vs_spp_spp.csv')
spp_compressed1['Agent'] = ['S++'] * len(spp_compressed1)
spp_compressed1.rename(columns={'S++': 'Rewards'}, inplace=True)

spp_compressed2 = pd.read_csv('./pennies_game/eee_vs_spp_spp.csv')
spp_compressed2['Agent'] = ['S++'] * len(spp_compressed2)
spp_compressed2.rename(columns={'S++': 'Rewards'}, inplace=True)

spp_compressed3 = pd.read_csv('./pennies_game/bbl_vs_spp_spp.csv')
spp_compressed3['Agent'] = ['S++'] * len(spp_compressed3)
spp_compressed3.rename(columns={'S++': 'Rewards'}, inplace=True)

algaater_compressed = pd.concat([algaater_compressed1, algaater_compressed2, algaater_compressed3], axis=0,
                                ignore_index=True)

bbl_compressed = pd.concat([bbl_compressed1, bbl_compressed2, bbl_compressed3], axis=0, ignore_index=True)

eee_compressed = pd.concat([eee_compressed1, eee_compressed2, eee_compressed3], axis=0, ignore_index=True)

spp_compressed = pd.concat([spp_compressed1, spp_compressed2, spp_compressed3], axis=0, ignore_index=True)

combined_df = pd.concat([algaater_compressed, bbl_compressed, eee_compressed, spp_compressed], axis=0,
                        ignore_index=True)
combined_df.to_csv('./pennies_game/combined_compressed_smart.csv')
combined_df.reset_index(drop=True, inplace=True)

plt.boxplot([algaater_compressed['Rewards'], bbl_compressed['Rewards'], eee_compressed['Rewards'],
             spp_compressed['Rewards']], labels=['AlgAATer', 'BBL', 'EEE', 'S++'])
plt.xlabel('Agent')
plt.ylabel('Rewards')
plt.title('Agent Rewards - Matching Pennies')
plt.savefig(f'./pennies_game/agent_rewards_smart.png', bbox_inches='tight')
plt.clf()
