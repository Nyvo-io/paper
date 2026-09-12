# Diffusion Policy：为什么预测噪声能生成多峰动作

> 本文只讨论一个问题：Diffusion Policy 为什么可以通过预测噪声，逐步生成符合专家分布的 action chunk。

为简化记号，下面用 $t$ 表示 diffusion noise level，用 $O$ 表示作为条件输入的 observation。定义

$$
\sigma_t = \sqrt{1-\bar\alpha_t}.
$$

## 训练过程

先随机取一个 $t$，再由专家动作序列 $A_0$ 和随机噪声构造 noisy action chunk $A_t$：

$$
A_t
=
\sqrt{\bar\alpha_t}A_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$

训练时，网络输入的是“被污染到第 $t$ 个噪声等级的 action chunk”、$t$ 以及 observation $O$；$t$ 告诉网络“现在污染得有多严重”。

推理时，初始 action chunk 是随机高斯噪声，但 $t$ 不是随机选取的，而是按采样器规定的顺序从最大噪声等级 $T$ 逐步走到 1。$t=T$ 表示 $A_T$ 处于噪声最强的等级。

最大等级 $T$ 由噪声日程设定。训练时，随机采样 $t\in\{1,2,\ldots,T\}$；$t$ 表示当前的 diffusion noise level。

diffusion steps 可以理解成：

> 从干净数据逐渐变成纯噪声，中间人为划分出来的 T 个等级

网络收到：

$$
(A_t,t,O)
$$

然后输出：

$$
\epsilon_\theta(A_t,t,O)
$$

训练目标就是让它不断接近刚才真正加入的那个噪声：

$$
\mathcal L(\theta)
=
\mathbb E
\left[
\left\|
\epsilon-
\epsilon_\theta(A_t,t,O)
\right\|^2
\right]
$$

最小化这个损失：对许多训练样本和随机噪声重复计算，再取平均。

网络输出的预测值其实是条件平均。假设固定某个 $(A_t,t,O)$，理论上可能存在许多 $(A_0,\epsilon)$ 组合都能产生这个 $A_t$。因此，面对同样的输入 $(A_t,t,O)$，训练数据中可能存在不同的正确噪声 $\epsilon^{(1)},\epsilon^{(2)},\epsilon^{(3)},\ldots$。

那么经过大量数据上的 MSE 训练，最优预测就是：

$$
\epsilon_\theta^*(A_t,t,O)
=
\mathbb E[
\epsilon\mid A_t,t,O
]
$$


最终训练出来一个函数 $\epsilon_\theta(A_t,t,O)$：

给我任何 noisy action、当前噪声等级和 observation，我都能判断它里面大概有哪些噪声成分。



## 推理过程

推理时，我们已经没有专家 $A_0$ 了。

机器人只知道当前 observation $O$。

我们先随机生成一整个 action chunk：$A_T\sim\mathcal N(0,I)$。

然后问网络：

- 这是 $A_T$；
- 现在 $t=T$；
- 机器人看到 $O$。

你觉得这里面的 noise 是什么？

网络输出：$\hat\epsilon_T$。

根据这个预测和采样器的更新规则，我们得到稍微干净一点的 $A_{T-1}$。

然后不断重复预测和去噪，直到得到 $A_0$，即机器人最终要执行的动作序列。


## 为什么预测 noise，而不是直接预测 $A_0$

第一，因为 **noise 是免费标签**。噪声 $\epsilon$ 是我们自己生成的，所以无需额外人工标注就知道正确答案。


第二，因为知道 noise，就能估计干净动作，并构造反向去噪更新。

$$
\hat A_0=
\frac{
A_t-\sqrt{1-\bar\alpha_t}\hat\epsilon
}{
\sqrt{\bar\alpha_t}
}
$$

在推理过程中，一点一点地去噪，得到最终的 $A_0$。

在最高噪声等级，$A_T$ 几乎没有动作信息，网络很难通过一次预测直接恢复精确动作。逐步去噪把生成过程拆成多个局部修正：

- 高噪声时只恢复粗略结构，例如选择左绕还是右绕；
- 中等噪声时形成动作轨迹；
- 低噪声时修正精确的位置和速度。


## MSE 为什么学到条件平均

下面的数学是在解决一个隐藏的问题：

> 网络通过 MSE 学到的，通常不是某一次加进去的真实噪声 $\epsilon$，而是条件平均
> $\mathbb E[\epsilon\mid A_t,O,t]$。
> 为什么这个平均值仍然能用于生成复杂、多峰的动作分布？


给定 $A_t$ 时，它不一定对应唯一的 $A_0$。

假设同一个观测 $O$ 下，专家有两种合理动作：

$$
A_0=-1 \quad\text{或}\quad A_0=+1
$$

实际场景中的 $A_0$ 通常是一整个动作向量序列。

它们分别加噪后，都有可能得到相同或非常接近的 $A_t$。

因此，给定网络输入 $(A_t,O,t)$，真实噪声未必唯一。我们之前得到的 MSE 最优预测是：

$$
\epsilon_\theta^*(A_t,O,t)
=
\mathbb E[\epsilon\mid A_t,O,t]
$$
也就是：

> 在当前这个 $A_t$、$O$、$t$ 下，所有可能 $\epsilon$ 的条件平均。


我们训练出来的是“平均 ε”，可是这个平均 ε 凭什么是正确的去噪方向？

把

$$
\hat\epsilon = \mathbb E[\epsilon\mid A_t,O,t]
$$

代入：

$$
\hat A_0 = \frac{ A_t-\sigma_t\mathbb E[\epsilon\mid A_t,O,t] }{ \sqrt{\bar\alpha_t} }
$$

根据加噪公式：

$$
\epsilon = \frac{ A_t-\sqrt{\bar\alpha_t}A_0 }{ \sigma_t }
$$

对条件分布取期望：

$$
\mathbb E[\epsilon\mid A_t,O,t] = \frac{ A_t-\sqrt{\bar\alpha_t} \mathbb E[A_0\mid A_t,O,t] }{ \sigma_t }
$$

代回上面的公式，得到：

$$
\boxed{ \hat A_0 = \mathbb E[A_0\mid A_t,O,t] }
$$

所以上面的公式配合 MSE 网络，算出来的并不一定是某个真实专家动作，而是：

> 在当前 $(A_t,O,t)$ 下，所有可能干净动作的条件平均。

这就是为什么“直接算一次 $\hat A_0$”还没有完整解决问题。

如果左右两种动作同样可能：

$$
A_0=-1 \quad\text{或}\quad A_0=+1
$$

在最模糊的位置，条件平均可能是：

$$
\mathbb E[A_0\mid A_t,O,t]=0
$$

但 0 可能恰好是撞上障碍物的动作。

所以 Diffusion Policy 选择逐步去噪。假设 $T=100$，那么：

$t=100$ 时，动作非常模糊；利用预测的噪声和对应的反向采样公式，可以得到噪声稍弱的 $A_{99}$。



## 条件平均为什么不会简单地把动作平均掉

问题变成：

> 把“平均噪声” (不是真正噪声) 代入去噪公式，为什么能正确生成数据？会不会只是把动作平均掉？

单靠反解公式回答不了这个问题，因为反解公式只有在 $\hat\epsilon=\epsilon$ 时才是严格的代数逆运算。

数学上可以证明：

$$
\nabla_{A_t}
\log p_t(A_t\mid O)
=
-\frac{1}{\sigma_t}
\mathbb E[\epsilon\mid A_t,O,t]
$$

而网络恰好学到：

$$
\epsilon_\theta(A_t,t,O)
\approx
\mathbb E[\epsilon\mid A_t,O,t]
$$

神经网络得到的就是这个期望。

所以这个条件期望经过尺度变换后给出了 score，也就是局部提高对数概率密度的方向。


说明网络虽然没有恢复某次真实加入的噪声，但它输出的是：

> 当前带噪动作应该往哪里移动，才能进入更高概率的动作区域。



然后反向扩散理论告诉我们：只要每个噪声等级都有这个正确的 score，就可以从高斯噪声逐步采样回专家动作分布。


$\log p_t(A_t\mid O)$ 的意思是：

给定机器人当前 observation $O$，在 diffusion 的第 $t$ 个噪声等级下，各种 noisy action chunk $A_t$ 的概率密度。

我们最终想得到的是 $p_{\text{data}}(A_0\mid O)$，也就是 $p_0(A_0\mid O)$。

推理过程是：

$$
p_T \longrightarrow p_{T-1} \longrightarrow \cdots \longrightarrow p_0
$$

每一步都用 score 构造从 $p_t$ 到 $p_{t-1}$ 的反向转移。score 提供局部方向，但完整采样不是单纯的梯度上升：更新还取决于噪声日程和采样器，并且在随机采样器中包含随机项。

因为这里的 $p_t(A_t\mid O)$ 不是随便一个概率分布，它本身就是“专家动作分布经过第 $t$ 级加噪后得到的分布”。


因此预测 noise 相当于学到了：

> **在当前 $A_t$ 所在的位置，应该朝哪个方向移动，才能更接近专家动作分布。**


之前训练就是加噪：

$$
p_0
\rightarrow
p_1
\rightarrow
p_2
\rightarrow
\cdots
\rightarrow
p_T
\approx
\mathcal N(0,I)
$$

前向过程中也可以用开头的公式从 $A_0$ 直接采样出任意等级的 $A_t$。

我们现在反过来：

从 $A_T\sim\mathcal N(0,I)$ 开始，
网络告诉我们当前 $A_T$ 对应的 score：

$$
\nabla_{A_T}\log p_T(A_T\mid O)
$$

这个方向告诉你：

> 在当前噪声等级 $T$ 下，往哪里走会进入 $p_T$ 更高概率的区域。

但是这里你要注意：

**我们不是找到 $p_T$ 的最高点之后就宣布“这是专家动作”。**

这不对。

$p_T$ 还是一个被严重加噪过的分布。

真正过程是：

$$
p_T \rightarrow p_{T-1} \rightarrow p_{T-2} \rightarrow \cdots \rightarrow p_0
$$

每一步都降低一点噪声。




## score 与条件期望的关系

数学推导如下：

$$
\nabla_{A_t}
\log p_t(A_t\mid O)
=
-\frac{1}{\sigma_t}
\mathbb E[\epsilon\mid A_t,O,t]
$$

如果固定 $A_0$，那么 $A_t$ 就是一个高斯随机变量：

$$
A_t\mid A_0
\sim
\mathcal N
\left(
\sqrt{\bar\alpha_t}A_0,\,
\sigma_t^2 I
\right)
$$

所以：

$$
q(A_t\mid A_0)
\propto
\exp
\left(
-\frac{
\|A_t-\sqrt{\bar\alpha_t}A_0\|^2
}{
2\sigma_t^2
}
\right)
$$


取 log：

$$
\log q(A_t\mid A_0)
=
C-
\frac{
\|A_t-\sqrt{\bar\alpha_t}A_0\|^2
}{
2\sigma_t^2
}
$$

这里 $C$ 是与 $A_t$ 无关的常数。

现在对 $A_t$ 求导：

$$
\nabla_{A_t}
\log q(A_t\mid A_0)
=
-
\frac{
A_t-\sqrt{\bar\alpha_t}A_0
}{
\sigma_t^2
}
$$

这个梯度就是所谓的 **score**。


而前面的加噪公式告诉我们：

$$
A_t-\sqrt{\bar\alpha_t}A_0
=
\sigma_t\epsilon
$$

最后得出：

$$
\nabla_{A_t}
\log q(A_t\mid A_0)
=
-\frac{\sigma_t\epsilon}{\sigma_t^2}
=
-\frac{\epsilon}{\sigma_t}
$$

也就是：

$$
\boxed{
\text{score}
=
-\frac{\epsilon}{\sigma_t}
}
$$

到这里还是 **q**，而且是：

> 已知具体专家动作 $A_0$ 时的 score。


这里 $q(A_t\mid A_0)$ 的意思是：

已知原始干净动作序列 $A_0$（专家动作），按照规定的加噪规则把它加到第 $t$ 级噪声后，$A_t$ 的概率分布是什么？


这里的 $A$ 不是单个动作，而是一整个 action chunk。


在 Diffusion Policy 中，控制时刻 $k$ 可以使用最近 $T_o$ 帧 observation：

$$
O_k=
[o_{k-T_o+1},\ldots,o_k]
$$

然后预测未来 $T_p$ 步动作：

$$
A_k=
[a_k,a_{k+1},\ldots,a_{k+T_p-1}]
$$

系统从中执行前 $T_a$ 步，再根据新 observation 重新规划。这里建模的是一整段动作发生的概率。


score 指的是：

$$
s(x)
=
\nabla_x\log p(x)
$$
核心是对 $x$ 求梯度。

它是在问：

> **如果我稍微改变当前的 x，概率会往哪个方向增加最快？**

沿着这个梯度方向做一个局部更新，会提高当前位置附近的对数概率密度；完整的反向扩散还需要按采样器执行每一级的转移。

现在是对 $A_t$ 求导：

我们现在有一个 noisy action $A_t$。

我们想知道：

> **这个 noisy action 应该往哪里移动，才能变得更像真实的专家动作？**




而：
$p_{\text{data}}(A_0\mid O)$ 是我们最终想学的东西：在机器人看到 observation $O$ 时，专家未来可能采取哪些 action chunks？

怎么得到它：

推理的时候，网络根本不知道真正的 $A_0$。

网络只拿到 $(A_t,t,O)$。

它预测 $\epsilon_\theta(A_t,t,O)$。

我们已知：

$$
\epsilon_\theta^*(A_t,t,O)
=
\mathbb E[
\epsilon
\mid
A_t,O,t
]
$$

$$
\epsilon
=
-\sqrt{1-\bar\alpha_t}
\nabla_{A_t}\log q(A_t\mid A_0)
$$

所以：

$$
\mathbb E[
\epsilon
\mid
A_t,O,t
]
=
-\sigma_t
\mathbb E
\left[
\nabla_{A_t}
\log q(A_t\mid A_0)
\mid
A_t,O,t
\right]
$$


然后有一个很重要的恒等式：

$$
\mathbb E
\left[
\nabla_{A_t}
\log q(A_t\mid A_0)
\mid
A_t,O,t
\right]
=
\nabla_{A_t}
\log p_t(A_t\mid O)
$$

所以：

$$
\mathbb E[
\epsilon
\mid
A_t,O,t
]
=
-\sigma_t
\nabla_{A_t}
\log p_t(A_t\mid O)
$$

而神经网络经过 MSE 训练后：

$$
\epsilon_\theta(A_t,t,O)
\approx
\mathbb E[
\epsilon
\mid
A_t,O,t
]
$$

所以最后：

$$
\boxed{
\epsilon_\theta(A_t,t,O)
\approx
-\sigma_t
\nabla_{A_t}
\log p_t(A_t\mid O)
}
$$

这里真正避免“把左右两种动作平均成中间动作”的关键，不是某一次 $\hat A_0$ 的条件均值本身，而是：推理从随机的 $A_T$ 出发，并在每个噪声等级使用局部 score 执行完整的反向采样。不同的随机初值，以及随机采样器中的随机转移，可以进入不同的高概率区域；即使使用确定性采样器，不同的随机初值也能导向不同模态。因此最终样本不必都收敛到全局均值。

## 参考

- [[Diffusion Policy - Visuomotor Policy Learning via Action Diffusion.pdf]]
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
