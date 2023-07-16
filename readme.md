Apologies, the code in this repo admittedly is not the greatest, but hopefully you can at least see how we implemented AlegAATr and conducted our experiments.

This repo contains the code we used for our second implementation ("Implementation II") of AlegAATr in the repeated games case study we performed.

The repo is broken into 5 key subdirectories within the generalized directory:
    
1 - agents: This subdirectory contains a python file for each agent we used, including AlegAATr.  There are also specific agents for each game that we used in our experiments.  The files subdirectory contains JSON policy files for CFR, folk egal, and minimax.

2 - analysis: All of the code we used for statistical tests, generating images, etc. is contained in this subdirectory.

3 - games: This subdirectory contains code for all of the games we used in our experiments.

4 - simulations: This subdirectory is large, as it contains a python file for each simulation we ran (AlegAATr vs. BBL on the block dilemma, EEE vs. S++ on the chicken game, BBL self play on the coordination game, etc.)

5 - training: Finally, this subdirectory contains the code we used to generate AAT trianing data for AlegAATr on each game.