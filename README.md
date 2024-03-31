# rl-asterix

Asterix is the name of an Atari 2600 game released in 1983, based on the comic book series of the same name. The game involves controlling Asterix or Obelix as they collect helmets dropped by Roman soldiers, while avoiding obstacles and enemies. The game has four levels of difficulty and two modes: single player or two players alternating. ¹

Asterix is also one of the Atari games that have been used as a benchmark for reinforcement learning algorithms. Reinforcement learning is a type of machine learning that involves learning from rewards and punishments. The goal is to train an agent to maximize its cumulative reward by interacting with an environment. Atari games provide a challenging and diverse environment for reinforcement learning, as they require high-dimensional sensory input, complex control policies, and long-term planning. ²

In this repository, we use PPO to build an RL agent to achieve high scores in the game. Some of the other reinforcement learning algorithms that have been applied to Asterix and other Atari games are:

- DQN: A deep Q-network that uses a convolutional neural network to learn a value function from raw pixels. DQN was the first algorithm to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. ³
- A3C: An asynchronous advantage actor-critic algorithm that uses multiple parallel workers to explore different parts of the environment and update a shared neural network. A3C combines the advantages of policy gradient methods and value-based methods, and achieves better performance than DQN on many Atari games. ⁴
- R2D2: A recurrent replay distributed DQN that uses a recurrent neural network to handle partial observability and a distributed prioritized experience replay to improve data efficiency and stability. R2D2 outperforms DQN and A3C on most Atari games, and achieves the highest score on Video Pinball. ⁵
- MuZero: A model-based reinforcement learning algorithm that learns a model of the environment dynamics and reward function from scratch, without any prior knowledge. MuZero uses a neural network to predict the next observation, reward, and value, given the current observation and action. MuZero achieves state-of-the-art performance on many Atari games, as well as board games and video games. 


¹: [AtariAge - Atari 2600 - Asterix (Atari)](^1^)
²: [Atari Games | Papers With Code](^2^)
³: [Playing Atari with Deep Reinforcement Learning](^3^)
⁴: [Asynchronous Methods for Deep Reinforcement Learning](^4^)
⁵: [Recurrent Experience Replay in Distributed Reinforcement Learning](^5^)
: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model]

Source: Conversation with Bing, 01/02/2024
(1) Atari Games | Papers With Code. https://paperswithcode.com/task/atari-games.
(2) [1312.5602] Playing Atari with Deep Reinforcement Learning - arXiv.org. https://arxiv.org/abs/1312.5602.
(3) Competitive Reinforcement Learning in Atari Games. https://link.springer.com/chapter/10.1007/978-3-319-63004-5_2.
(4) Model Based Reinforcement Learning for Atari. https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2021_2022/papers/Kaiser_ArXiv_2019.pdf.
(5) undefined. https://doi.org/10.48550/arXiv.1312.5602.
(6) Atari Games | Papers With Code. https://paperswithcode.com/task/atari-games.
(7) [1312.5602] Playing Atari with Deep Reinforcement Learning - arXiv.org. https://arxiv.org/abs/1312.5602.
(8) Competitive Reinforcement Learning in Atari Games. https://link.springer.com/chapter/10.1007/978-3-319-63004-5_2.
(9) Model Based Reinforcement Learning for Atari. https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2021_2022/papers/Kaiser_ArXiv_2019.pdf.
(10) undefined. https://doi.org/10.48550/arXiv.1312.5602.
