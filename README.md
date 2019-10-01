# kArmedBandit
Basic reinforcement learning algorithms to solve the k-Armed Bandit Problem, based on Barto and Sutton.


This project was completed in the summer of 2018, before my freshman year of college. It implements some of the basic reinforcement learning algorithms to solve the k-Armed Bandit Problem. I gradually added to it as I read through the first few chapters of Sutton and Barto's "Reinforcement Learning."  

A "Bandit" is essentially a slot machine; the agent can choose one out of k possible bandits, and each bandit gives a random reward with a different average; some Bandits are more rewarding than others, on average. The goal of the agent is to maximize it's total reward, and in order to do that it must learn what the average reward of each Bandit is. 

The k-Armed Bandit problem is often used to showcase the explore/exploit tradeoff. The agent can either try gather new information (explore) or act 'optimally' based on its current information (exploit). 

This program implements 3 basic approaches to this problem:

Epislon greedy - Every time you pull a lever and get a reward from a bandit, update the average reward of that bandit. When chosing a lever, some (small) percent of the time, called epsilon, you choose a random lever with equal probability for each lever. The rest of the time, you choose the lever with the highest reward. 

Weighted - Probabilty of selecting a given bandit is proportional to its value divided by the total value. The "value" is updated to reflectt the current average, as with e-greedy.

Gradient - When you choose a lever, shift the probabilities of selecting that lever in the direction of of reward you recived. 
