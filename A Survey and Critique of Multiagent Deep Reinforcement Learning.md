***********************
这是2019年的一篇关于多智能体深度强化学习的综述。本文提供了当前多智能体深度强化学习（Multiagent deep reinforcement learning，MDRL）的概述。强化学习（Reinforcement learning，RL）的火热导致了一系列应用和算法的涌现，应用也从单智能体场景扩展到多智能体场景（Multiagent learning，MAL）中。

主要内容：
1. MAL和RL中的关键内容，重点介绍它们如何应用到MDRL中
2. 介绍benchmark和研究的方法
3. 介绍MDRL目前面临的挑战
***********************
 
# 1.介绍
多智能体学习中的技术主要来自两个领域：强化学习（RL）和深度学习（DL）

在强化学习中智能体通过与动态环境的交互学习，但是RL与传统机器学习一样，依赖于人工设计特征。而深度学习通过神经网络（neaural networks，NN）来自动发现高维数据中的表征，两种技术结合起来就是深度强化学习（Deep reinforcement learning，DRL）。DRL被认为是走向通用AI系统的一项重要技术。该技术融入多智能体系统后，出现了新兴的研究领域：多智能体深度强化学习（MDRL）。
+ MDRL研究难点
  + 非平稳性
  + 维度灾难
  + 智能体信誉分配
  + 全局探索
  + 过度泛化
+ MDRL分类
  + 行为分析：在多智能体场景中评估单智能体DRL算法
  + 学习通信：智能体学习通信协议来解决合作任务
  + 学习合作：智能体仅使用动作和观测来学习合作
  + 智能体建模：智能体对其它智能体进行推理以完成任务

# 2.单智能体学习
## 2.1强化学习（RL）
马尔可夫决策过程（Markov decision process，MDP）是单智能体完全可观环境中获得最优决策的一个合适的模型。一个MDP通过元组$\langle\mathcal{S}, \mathcal{A}, R, T, \gamma\rangle$来定义。
+ Q-learning算法
  + 最常见的RL算法，适用条件：平稳环境、单智能体、离散动作的完全可观测环境
  + 通过以下公式更新$\hat{Q}(s, a) \leftarrow \hat{Q}(s, a)+\alpha\left[\left(r+\gamma \max _{a^{\prime}} \hat{Q}\left(s^{\prime}, a^{\prime}\right)\right)-\hat{Q}(s, a)\right]$
+ REINFORCE算法（蒙特卡洛策略梯度）
  + 策略梯度的方法不需要价值估计作为中继而直接学习参数化策略
  + REINFORCE通过梯度下降更新参数，蒙特卡洛方法（MC）对整个轨迹采样
  + 参数更新公式$\theta_{t+1}=\theta_{t}+\alpha G_{t} \frac{\nabla \pi\left(A_{t} ; S_{t}, \theta_{t}\right)}{\pi\left(A_{t} ; S_{t}, \theta_{t}\right)}$
  + 策略梯度方法有较高的方差，后续可以可以通过引入baseline来缓解
+ actor-critic算法（价值方法和策略方法的结合），在多智能体场景中方差进一步扩大
  + 其中actor代表策略，来进行动作的选择；critic来学习价值函数
  + AC算法代表：DPG，critic使用Q-learning，actor梯度更新，该算法后被扩展至DRL和MDRL
+ 在MDRL中解决高方差的算法：COMA、MADDPG

## 2.2深度强化学习（DRL）
表格型的强化学习方法很难解决复杂场景下的问题，因此引入深度学习技术，升级网络（NN）作为函数近似器来解决。
+ 优点
  + 泛化性强，提高了大状态空间的样本效率
  + 减少甚至避免手动设计状态信息的表征
+ 缺点
  + 监督学习中要求数据是独立同分布的，RL中训练数据涉及到环境交互不满足独立性条件
  + RL训练数据是非平稳的

一般来说，策略梯度方法比值函数方法具有更强的收敛保证！！！

简要介绍几种算法：
+ 基于价值的方法：DQN
  + DQN使用神经网络来拟合价值函数；使用经验缓冲池（experience replay,ER）来存储交互数据，有助于缓解训练的非平稳性，其主要应用于离线的RL方法
  + DRQN将DQN推广到部分可观的环境中，使用递归神经网络（LSTMs）
  + Double DQN减轻过高估问题
  + dueling-DQN分解Q函数，分为两个数据流一个学习状态价值另一个学习优势函数
  + ![](DQN.png)
+ 策略梯度方法：针对高维连续动作，DDPG
  + DDPG是无模型的离线AC算法，基于DPG算法
  + 使用了batch normalization来确保泛化性
+ A3C（Asynchronous Advantage Actor-Critic）
  + 并行异步训练方案，使用多线程
  + 在线RL方法，本地计算梯度，全局网络优化
  + loss包含策略loss（actor）和价值loss（critic）
  + 参数更新使用优势函数：$A\left(s_{t}, a_{t} ; \theta_{v}\right)=Q(s, a)-V(s)$
  + ![](A3C.png)
+ A2C(Advantage Actor-Critic)
  + 多线程同时更新全局神经网络
+ ACER（Actor-critic with experience replay）
  + A3C架构、离线学习
  + 使用经验回放池
+ IPG（Interpolated Policy Gradient）
  + ACER和DDPG结合
+ UNREAL框架（Unsupervised Reinforcement and Auxiliary Learning）
  + 在A3C架构之上引入无监督的辅助任务
  + 使用优先ER池（给予高奖励更高的采样概率）
+ IMPALA（Importance Weighted Actor-Learner Architecture）
  + 分布式架构
  + actor将经验轨迹（trajectories）传递给集中的学习者，动作与学习分离
+ TRPO（Trust Region Policy Optimization）和PPO（Proximal Policy Optimization）
  + 二者都是比较新技术，PPO更是效果佳
  + PPO通过损失函数防止训练时策略的突变
  + 可以分布式使用DPPO（Distributed PPO）
+ 熵正则化RL框架（entropy-regularized RL）
  + 策略梯度算法+Qlearning
+ SAC（Soft Actor-Critic）
  + 学习一个随机策略，两个Q函数和一个价值函数
  + 当前策略收集数据和经验回访交替进行

# 3.多智能体深度强化学习（MDRL）
整体框架+不同研究方向的研究
## 3.1多智能体学习
多智能体环境比单智能体的更加复杂，因为智能体需要同时与环境和其余智能体交互。
+ 独立学习者（independent learner）
  + 直接将单智能体算法放入多智能体环境中
  + 算法假设被打破，智能体各自学习策略而将其余智能体看作环境
  + 环境不再是稳定的，因此马尔可夫特性变得无效
  + 训练可能会失败，但是实践中会有应用，有可扩展性（效果可能还行！）

多智能体（n个智能体）角度的MDP主要不同：
+ n维的action
+ n维的reward
+ n维的状态转移矩阵

智能体的最优策略依赖于其余智能体的策略，其余智能体是非平稳的

收敛性结果：
+ 敌对环境（零和游戏）可以保证针对任意对手的最优策略，极大极小Q学习（Minimax Q-learning）
+ 合作环境中，需要对其他智能体有强假设，以确保收敛到最优行为（eg. Nash Q-learning && Friend-or-Foe Q-learning）
+ 其他环境已知没有保证收敛的基于值的RL算法

MAL中其他常见问题
+ 动作阴影（action shadowing）
+ 维度诅咒（爆炸）
+ 多智能体信用分配

MAL中研究方向
+ 博弈论和MARL
+ 合作问题
+ 多智能体学习的进化动力学
+ 动态环境中的学习
+ 智能体建模
+ RL中的迁移学习

## 3.2多智能体深度强化学习分类
算法可能属于多个类别，类别之间不互斥
+ 行为分析：主要在多智能体环境中分析评估DRL算法
  + 合作场景
  + 竞争场景
  + 混合场景（二者结合）
+ 学习通信：智能体通过通信协议来分享信息
+ 学习合作（一般用于合作、混合场景）
+ 智能体建模（一般用于竞争、混合场景）
  + 合作
  + 对手建模
  + 目标推断
  + 智能体行为解释

## 3.3行为分析
+ 早期，通过调整奖励函数来导致合作或竞争的自然行为
+ 自博弈：容易忘记过去的知识，收敛到纳什均衡点
+ 探索奖励：密集的奖励、提高样本效率、提供对手采样

## 3.4通信学习
协作场景，智能体通过共享信息来最大化共享效用

基于通信的算法：
+ RIAL：单个网络（参数共享）
+ DIAL：学习和通信过程中通信信道传递梯度
+ CommNet：单一网络中使用向量通道，多个通信周期，智能体动态变化
+ BiCNet：AC架构，通信在潜在空间中发生、参数共享、不显示通信
+ MD-MADDPG：共享内存
+ MADDPG-MD：直接通信场景，使用dropout提高通信可靠性，鲁棒性，方便虚拟和实际的迁移。

## 3.5合作学习
基于合作的算法：
+ 很多通信的算法也可以算作合作这一类，主要是通过通信来促进合作
+ DCH/PSRO：计算混合策略的近似最佳响应
+ Fingerprints：处理MDRL中的ER问题，ER工作的前提是数据分布满足某种假设，解决方案是向ER中加入信息，帮助消除采样数据的滞后性。
+ Lenient-DQN：容忍低reward的行为，实现宽容乐观合作，ER问题解决方案是向ER中加入信息
+ （DEC）Hysteretic-DRQN：两个学习率、策略蒸馏、多任务学习、完全分布式
+ WDDQN：宽容、双估计器、优先体验重放缓冲区
+ FTW：混合环境、两级架构和基于群体的学习，Q学习的层次结构（slow RNN+fast RNN 不同时间尺度）、ELo系统
+ MADDPG：存在问题-智能体数量的增加导致高方差和梯度的错误概率，分散执行的集中策略
+ VDN：整体行动价值分为多个智能体的部分
+ QMIX：团队行动价值函数分解，非线性重新组合它们的混合网络
+ COMA：解决多智能体信用分配、中心式critic和反事实优势函数（差异报酬），完全集中式的方法不存在非平稳性，但难以扩展
+ PS-DQN，PS-TRPO，PS-A3C，MADDPG：参数共享、AC方法、所有智能体行为更新critic，其中PS-PRPO表现最好

完全集中式的方法不存在非平稳性，但难以扩展；独立学习的智能体适合规模化，但存在非平稳性问题。
## 3.6智能体建模
不完全信息博弈通常需要随机策略来实现最佳行为。纳什均衡在MAL算法中的探索：Nash-Q learning和Minimax-Qlearning

智能体建模：
+ MADDPG：AC方法、所有智能体行为更新critic
+ DRON：早期工作，DQN推断对手行为；专家混合思想，门控网络决定专家选择
+ SOM：两个网络一个计算自己的策略，另一个推断对手的目标
+ DPIQN、DPIRQN、SOM：辅助任务学习策略特征、奖励依赖于两个智能体的目标，智能体通过本身策略推断另一个智能体
+ NFSP：自博弈+两个NN计算纳什均衡；使用平均和最佳响应网络混合进行行为
+ M3DDPG：极大极小目标扩展MADDPG；隐式的采用minmax
+ LOLA：利用学习规则，考虑别的智能体参数更新
+ ToMnet：端到端对手学习推理；目标是对手下一步行动
+ Deep Bayes-ToMop：贝叶斯策略重用、思维评论
+ Deep BPR+：贝叶斯重用和策略蒸馏

# 4.RL、MAL、MDRL三者关系
## 4.1MDRL中的一些例子
+ 处理独立学习者的非平稳性：Hyper-Q、fingerprint、LDQNs、DEC-HDRQNS
+ 多智能体信用分配：COMA智能体优势函数
+ 多任务学习：涉及到策略蒸馏
+ 辅助任务：eg.DPIQN、DRPIO，辅助任务能有效消除局部极小值
+ 经验回放：使用前提条件包含环境是平稳的，多智能体场景往往会破坏这中前提
+ 双估计器：double DQN这种思想也应用于MDRL
## 4.2实践tips
+ 经验回放池：很多工作中向元组添加信息
+ 中心式训练分布式执行：在学习中使用附加信息（全局状态 动作等）在执行中删除该信息
+ 参数共享：训练单个网络，智能体共享权重
+ RNN网络：RNN变体LSTM和GRU解决了其长期依赖性效率低下的问题，循环网络在MDRL中解决部分可观等问题。
+ MAL中的过拟合：智能体易陷入局部最优，解决方案是拥有混合策略或更鲁棒的算法
## 4.3MDRL中的基准
+ ALE和openai Gym 单智能体RL
+ CMOTPS：两个智能体协同
+ Apprentice Firemen Game
+ Pommerman：合作对抗和混合环境
+ SMAC（Starcraft Multiagent Challenge）：包含QMIX和COMA 细粒度控制
+ MARLO
+ Hanabi
+ Arena:基于UE，内置independent PPO
+ MuJoCo Multiagent Soccer：三维动作空间 2V2
+ Neural MMO

## 4.4MDRL中的实际挑战
+ 可重复性、只选取最好结果
+ 通过调参才能使算法发挥作用，参数的调整耗费计算资源
+ MDRL中需要大量的计算资源

## 4.5开放性问题
+ 稀疏和延迟奖励：分层学习、手动设计密集奖励
+ 自博弈：简单的自博弈不会产生最佳结果，而进化方法效果较好，缺点是需要较大的计算资源
+ 组合挑战：eg.蒙特卡洛树搜索，搜索并行化

# 5.结论
深度强化学习在多智能体场景面临更大的困难；DRL在MDRL中相关工作很多，但还是有很多开放性的问题。