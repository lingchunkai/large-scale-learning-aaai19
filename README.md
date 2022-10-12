# Large scale learning of agent rationality in two-player zero-sum games
This repo contains the code for reproducing experiments for [large scale learning of agent rationality in two-player zero-sum games](https://arxiv.org/abs/1903.04101).

This paper is an extension of prior work on [differentiably learning 2-player zero-sum games](https://arxiv.org/abs/1805.02777), whose code can be found [here](https://github.com/lingchunkai/payoff_learning). 

The structure of the code is the same as before, with primary differences in the FOM methods used to speed up the forward/backward passes, as well as gradients included for the lambda/temperature parameters used. The human data can be downloaded from the [Hunt. et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2000638).

