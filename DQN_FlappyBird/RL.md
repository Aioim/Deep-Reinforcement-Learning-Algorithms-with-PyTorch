[toc]

# Basic

状态值函数（State Value Function）$V^{\pi}(s)$ 表示从状态 $s$ 开始，执行策略 $\pi$ 得到的期望总回报。
$$
V^{\pi}(s)=\mathbb{E}_{\tau \sim p(\tau)}\left[\sum_{t=0}^{T-1} \gamma^{t} r_{t+1} | \tau_{s_{0}}=s\right]
$$
状态-动作值函数（State-Action Value Function）$Q^{\pi}(s, a)$ 表示初始状态为 $s$ 并进行动作 $a$，然后执行策略 $\pi$ 得到的期望总回报。
$$
Q^{\pi}(s, a)=\mathbb{E}_{s^{\prime} \sim p\left(s^{\prime} | s, a\right)}\left[r\left(s, a, s^{\prime}\right)+\gamma V^{\pi}\left(s^{\prime}\right)\right]
$$

# Temporal-Difference Learning  

时序差分学习方法结合了动态规划算法和蒙特卡罗方法，比仅仅使用蒙特卡罗采样方法的效率要高很多。时序差分学习方法是模拟一段轨迹，每行动一步 (或者几步)，就利用贝尔曼方程来评估行动前状态的价值。当时序差分学习方法中每次更新的动作数为最大步数时，就等价于蒙特卡罗方法。

将 $Q$ 函数的估计 $\hat{Q}$ 改为增量计算的方式：
$$
\begin{aligned}
\hat{Q}_{N}^{\pi}(s, a) &=\frac{1}{N} \sum_{n=1}^{N} G\left(\tau_{s_{0}=s, a_{0}=a}^{(n)}\right) \\
&=\frac{1}{N}\left(G\left(\tau_{s_{0}=s, a_{0}=a}^{(N)}\right)+\sum_{n=1}^{N-1} G\left(\tau_{s_{0}=s, a_{0}=a}^{(n)}\right)\right) \\
&=\frac{1}{N}\left(G\left(\tau_{s_{0}=s, a_{0}=a}^{(N)}\right)+(N-1) \hat{Q}_{N-1}^{\pi}(s, a)\right) \\
&=\hat{Q}_{N-1}^{\pi}(s, a)+\frac{1}{N}\left(G\left(\tau_{s_{0}=s, a_{0}=a}^{(N)}\right)-\hat{Q}_{N-1}^{\pi}(s, a)\right)
\end{aligned}
$$
值函数 $\hat{Q}^{\pi}(s, a)$ 在第 $N$ 次试验后的平均等于第 $N - 1$ 次试验后的平均加上一个增量。更一般性地，我们将权重系数 $\frac{1}{N}$ 改为一个比较小的正数 $\alpha$。
$$
\hat{Q}^{\pi}(s, a) \leftarrow \hat{Q}^{\pi}(s, a)+\alpha\left(G\left(\tau_{s_{0}=s, a_{0}=a}\right)-\hat{Q}^{\pi}(s, a)\right)
$$
$G\left(\tau_{s_{0}=s, a_{0}=a}\right)$ 为一次试验的完整轨迹所得到的总回报。为了提高效率，可以借助动态规划的方法来计算。从 $s, a$ 开始，采样下一步的状态和动作 $s′, a′$，并得到奖励 $r(s, a, s′)$，然后利用贝尔曼方程来近似估计 $G\left(\tau_{s_{0}=s, a_{0}=a}\right)$。  
$$
\begin{aligned}
G\left(\tau_{\left.s_{0}=s, a_{0}=a, s_{1}=s^{\prime}, a_{1}=a^{\prime}\right)}\right.&=r\left(s, a, s^{\prime}\right)+\gamma G\left(\tau_{s_{0}=s^{\prime}, a_{0}=a^{\prime}}\right) \\
& \approx r\left(s, a, s^{\prime}\right)+\gamma \hat{Q}^{\pi}\left(s^{\prime}, a^{\prime}\right)
\end{aligned}
$$
结合以上两个公式可得：
$$
\hat{Q}^{\pi}(s, a) \leftarrow \hat{Q}^{\pi}(s, a)+\alpha\left(r\left(s, a, s^{\prime}\right)+\gamma \hat{Q}^{\pi}\left(s^{\prime}, a^{\prime}\right)-\hat{Q}^{\pi}(s, a)\right)
$$
更新 $\hat{Q}^{\pi}(s, a)$ 只需要知道当前状态 $s$ 和动作 $a$、奖励 $r(s, a, s′)$、下一步的状态 $s′$  和动作 $a′$。这种策略学习方法称为 SARSA 算法（State Action Reward State Action，SARSA）。

SARSA 算法是 on-policy 的。Q-Learning 算法是 off-policy 的，用来执行策略和用来优化的不是同一个Q。
$$
Q(s, a) \leftarrow Q(s, a)+\alpha\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)-Q(s, a)\right)
$$
在算法中，最后一步更新 targetQ = reward，而不是targetQ=reward+γ*nextQ

# DQN

用神经网络近似值函数Q：
$$
Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) \approx Q^{\pi}(s, a)
$$
需要学习一个参数 ${\phi}$ 来使得函数 $Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})$ 可以逼近值函数 $Q^{\pi}(s, a)$。如果采用蒙特卡罗方法，就直接让 $Q^{\pi}(s, a)$ 去逼近平均的总回报 $\hat{Q}^{\pi}(s, a)$；如果采用时序差分学习方法，就让  $Q^{\pi}(s, a)$ 去逼近 $\mathbb{E}_{\boldsymbol{s}^{\prime}, \boldsymbol{a}^{\prime}}\left[r+\gamma Q_{\phi}\left(\boldsymbol{s}^{\prime}, \boldsymbol{a}^{\prime}\right)\right]$。

以 Q-Learning 为例，目标函数为：
$$
\mathcal{L}\left(s, a, s^{\prime} | \phi\right)=\left(r+\gamma \max _{a^{\prime}} Q_{\phi}\left(s^{\prime}, a^{\prime}\right)-Q_{\phi}(s, a)\right)^{2}
$$
存在两个问题：

- 目标不稳定，参数学习的目标依赖于参数本身。解决方法是目标网络冻结（Freezing Target Networks）
- 样本之间有很强的相关性。解决方法是经验回放（Experience Replay），即构建一
  个经验池（Replay Buffer）来去除数据相关性。

# Policy Gradient

直接对目标函数 $\mathcal{J}(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}[G(\tau)]$ 进行梯度上升。
$$
\begin{aligned}
\frac{\partial \mathcal{J}(\theta)}{\partial \theta} &=\frac{\partial}{\partial \theta} \int p_{\theta}(\tau) G(\tau) d \tau \\
&=\int\left(\frac{\partial}{\partial \theta} p_{\theta}(\tau)\right) G(\tau) d \tau\\

&=\int p_{\theta}(\tau)\left(\frac{1}{p_{\theta}(\tau)} \frac{\partial}{\partial \theta} p_{\theta}(\tau)\right) G(\tau) d \tau \\
&=\int p_{\theta}(\tau)\left(\frac{\partial}{\partial \theta} \log p_{\theta}(\tau)\right) G(\tau) d \tau \\
&=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\frac{\partial}{\partial \theta} \log p_{\theta}(\tau) G(\tau)\right]

\end{aligned}
$$
其中 $\frac{\partial}{\partial \theta} \log p_{\theta}(\tau)$ 可以分解为：
$$
\begin{aligned}
\frac{\partial}{\partial \theta} \log p_{\theta}(\tau) &=\frac{\partial}{\partial \theta} \log \left(p\left(s_{0}\right) \prod_{t=0}^{T-1} \pi_{\theta}\left(a_{t} | s_{t}\right) p\left(s_{t+1} | s_{t}, a_{t}\right)\right) \\
&=\frac{\partial}{\partial \theta}\left(\log p\left(s_{0}\right)+\sum_{t=0}^{T-1} \log \pi_{\theta}\left(a_{t} | s_{t}\right)+\sum_{t=0}^{T-1} \log p\left(s_{t+1} | s_{t}, a_{t}\right)\right)\\
&=\sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)
\end{aligned}
$$
梯度仅与策略函数相关。策略梯度可以写为：
$$
\begin{aligned}
\frac{\partial \mathcal{J}(\theta)}{\partial \theta} &=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\left(\sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right) G(\tau)\right] \\
&=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\left(\sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right)\left(G\left(\tau_{0: t-1}\right)+\gamma^{t} G\left(\tau_{t: T}\right)\right)\right] \\
&=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T-1}\left(\frac{\partial}{\partial \theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) \gamma^{t} G\left(\tau_{t: T}\right)\right)\right]
\end{aligned}
$$
过去的 reward 在理论上不影响策略梯度，为了减少观测误差，实际上应该去掉过去的 reward，证明如下：

>  https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof1.html

# DDPG

结合了 TD Learning 和 Policy Gradient，

采用类似 actor-critic 的架构，Actor Network 和 Critic Network 需要复制成两份，类似 DQN 的 fixed network 操作。

**Critic**

- Critic网络的作用是预估 **Q**，虽然它还叫Critic，但和AC中的Critic不一样，这里预估的是Q不是V；
- Critic的输入有两个：动作和状态，需要一起输入到Critic中；
- Critic 网络的 loss 其还是和 DQN 一样，用的是 TD-error。

**Actor**

- 和AC不同，Actor输出的是一个动作的连续值而不是离散值（多个动作的概率）；
- Actor的功能是，输出一个动作A，这个动作A输入到Crititc后，能够获得最大的Q值。
- 所以Actor的更新方式和AC不同，不是用带权重梯度更新，而是用梯度下降最小化 Critic 的输出值。

Target network soft update：在更新目标网络时，使用加权平均的更新法。

Noise：对 action 加入噪声，增加探索

感觉和 DQN 更像。Policy Gradient 的部分指的是可以对 Actor 进行优化，使之能够获得最大的Q值。