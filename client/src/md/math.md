---
title: Math Behind the Models
author: Jack Chen and Andrew Liu
date: January 9, 2023
time: 30 min
---

## Summary of math

Coming into this project, we wanted to have a solid understanding of the math underlying the model. We reproduce some of the things we learned here. Since we had some background knowledge, we'll only rigorously rederive results that we learned or found very interesting.

<a id="ddpm"></a>

### 1. DDPM

We begin with DDPMs [(Ho et al. 2020)](https://arxiv.org/abs/2006.11239). These are the foundation of all good diffusion models. In particular, InstructPix2Pix is a fine-tuned version of Stable Diffusion, which works out of the box with DDPM.

#### 1.1. Foundation

In these diffusion models, we noise images sampled from distribution $q(x)$ through the _forward process_, defined via

$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})\qquad q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I),
$$

where $$\beta_t$$ is some noise scheduler. We'll typically see $$\beta_1 < \beta_2 < \cdots < \beta_T$$ to ensure that the final image is pure noise.

Let $$\alpha_t = 1-\beta_t$$ and $$\prod_{i=1}^t \alpha_t = \overline{\alpha}_t$$. It can be shown that
<a id="forward-sampling"></a>

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\overline{\alpha}_t}x_0, (1-\overline{\alpha}_t)I),
$$

which not only makes sampling the forward process very efficient, but also shows that we can think of the forward process as some linear combination of the original image and pure noise.

In the _backward process_, the goal is to produce a model $$p_{\theta}$$ that approximates $$q$$, where we define

$$
p_{\theta}(x_{0:T}) = p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t) \qquad p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t)).
$$

Given perfect $$p_{\theta}$$, we can perfectly recreate the original data distribution from pure noise, which is the magic of diffusion. It turns out that $$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}(x_t,x_0), \tilde{\beta}_tI)$$ is tractable, and it can be shown that

$$
\tilde{\mu}(x_t,x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\varepsilon_t\right),
$$

where $$\varepsilon_t$$ is the noise added to produce $$x_t$$ from $$x_0$$. This quantity is relevant because it can be shown that optimizing the [variational lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) is equivalent to minimizing

$$
L = \mathbb{E}_{x_0, \varepsilon, t}\left[\frac{1}{2\lVert \Sigma_{\theta}(x_t,t)\rVert_2^2}\lVert \tilde{\mu}_t(x_t,x_0) - \mu_{\theta}(x_t,t)\rVert^2\right].
$$

where $$\mu_{\theta}(x_t,t)$$ and $$\Sigma_{\theta}(x_t,t)$$ is the backwards mean and variance predicted by our model. Substituting values for $$\tilde{\mu}(x_t, x_0)$$, this is equivalent to

<a id="ddpm-math-loss"></a>

$$
L = \mathbb{E}_{x_0,\varepsilon, t}\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\overline{\alpha}_t)\lVert\Sigma_{\theta}(x_t,t)\rVert_2^2}\lVert \varepsilon_t - \varepsilon_{\theta}(x_t,t)\rVert^2\right].
$$

In their paper, [(Ho et al. 2020)](https://arxiv.org/abs/2006.11239) showed that ignoring the constant in front can produce better results for training, so eventually reduced the loss function to

$$
L = \mathbb{E}_{x_0, \varepsilon, t}[\lVert \varepsilon_t - \varepsilon_{\theta}(x_t,t)\rVert^2].
$$

The full derivation of all the facts leading up to this point can be found [here](https://azliu0.github.io/mit-notes/6.7900/6_7900_Notes.pdf). The importance of this fact is that training DDPM models boils down to sampling random images at random timesteps, and having a model (typically a UNet) learn to predict the the noise added at timestep $$t$$.

#### 1.2. Clarifying $$x_t$$

The input of our UNet is an image $$x_t$$, generated from $$x_0$$ using the distribution $$q(x_t|x_0)$$. the output of our UNet is the noise used to generate $$x_t$$, $$\varepsilon_{\theta}(x_t,t)$$.

This noise is sampled from $$\mathcal{N}(0,1)$$, per the [reparamaterization trick](https://en.wikipedia.org/wiki/Variational_autoencoder#Reparameterization). this noise represents normalized quantity of noise added from $$x_0$$ to $$x_t$$, **not** from $$x_{t-1}$$ to $$x_t$$. in other words, this noise corresponds to the forwards distribution $$q(x_t|x_0)$$, and not $$q(x_t|x_{t-1})$$.

Even though we are predicting noise from $$x_0$$ to $$x_t$$, this noise is used to generate the distribution $$p(x_{t-1}|x_t)$$. in other words, the distribution for predicting one timestep backwards from $$t$$ to $$t-1$$ is a function of noise added from $$0$$ to $$t$$. this is because the original (tractable) distribution that we are trying to learn is $$q(x_{t-1}|x_t,x_0)$$, which also has access to information about $$x_0$$.

A concrete demonstration of this fact is the way that we simulate the backwards process once we have trained our UNet. We must make $$T$$ calls to the UNet to go from pure noise to the original data distribution (more accurately, our approximation of the distribution). In each of these $$T$$ calls, although the UNet estimates noise from timestep $$0$$ to the current timestep, we cannot jump directly to the beginning, because we do not know $$p(x_0|x_t)$$ as a function of this noise. Thus, although we have a shortcut for the forwards process, there is no equivalent shortcut for the backwards process, and so inference is quite expensive.

### 2. Multivariate Gaussians

Before continuing, we first derive some useful identities for multivariate Gaussians that will be used later. Many of these identities and derivations can be found in Pattern Recognition and Machine Learning [(Bishop 2006)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf). They are recreated here for completeness and for our own learning.

Suppose we represent Gaussian $$p(x)$$ as some jointly defined distribution $$p(x_a,x_b)$$, where $$x_a$$ and $$x_b$$ arbitrarily partition the dimensions in $$x$$. $$x_a$$ and $$x_b$$ are distributions in their own right, with means and covariances; write

$$
x = \begin{pmatrix}x_a \\ x_b\end{pmatrix}\qquad \mu = \begin{pmatrix}\mu_a \\ \mu_b\end{pmatrix}\qquad \Sigma = \begin{pmatrix}\Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb}\end{pmatrix}.
$$

Also, let

$$
\Lambda \equiv \Sigma^{-1} = \begin{pmatrix}\Lambda_{aa} & \Lambda_{ab} \\ \Lambda_{ba} & \Lambda_{bb}\end{pmatrix}
$$

be the precision matrix corresponding to $$x$$.

<a id="2.1"></a>

#### 2.1. Computing marginal gaussian from joint gaussian

The first question we will focus on is how to compute the marginal

$$
p(x_a) = \int p(x_a,x_b) \mathrm{d}x_b.
$$

The purpose of decomposing everything into $$x_a$$ and $$x_b$$ components is that we may now write
<a id="og-expansion"></a>

$$
\begin{align*}
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) &= -\frac{1}{2}(x_a-\mu_a)^T\Lambda_{aa}(x_a-\mu_a)-\frac{1}{2}(x_a-\mu_a)^T\Lambda_{ab}(x_b-\mu_b) \\
&\quad -\frac{1}{2}(x_b-\mu_b)^T\Lambda_{ba}(x_a-\mu_a)-\frac{1}{2}(x_b-\mu_b)^T\Lambda_{bb}(x_b-\mu_b).
\end{align*}
$$

Our strategy will be to isolate the integrating variable $$x_b$$. Collecting all terms in the above that have $$x_b$$:

$$
-\frac{1}{2}x_b^T\Lambda_{bb}x_b + x_b^T(\Lambda_{bb}\mu_b - \Lambda_{ba}x_a + \Lambda_{ba}\mu_a),
$$

where we use the fact that $$\Lambda_{ab} = \Lambda_{ba}^T$$ to combine like terms. After completing the square, our integral becomes

$$
\begin{align*}
f(x_a) \cdot \int & \exp\biggl\{-\frac{1}{2}(x_b-\Lambda_{bb}^{-1}(\Lambda_{bb}\mu_b - \Lambda_{ba}x_a + \Lambda_{ba}\mu_a))^T\\
&\quad \cdot \Lambda_{bb}(x_b-\Lambda_{bb}^{-1}(\Lambda_{bb}\mu_b - \Lambda_{ba}x_a + \Lambda_{ba}\mu_a))\biggr\}\mathrm{d}x_b,
\end{align*}
$$

where $$f(x_a)$$ is some function of $$x_a$$ independent of $$x_b$$. There are two key observations here:

- $$f(x_a)$$ is a quadratic form in $$x_a$$
- the integrand is just an unnormalized Gaussian, so it will integrate to the inverse normalization factor. in this case, this normalization factor is only a function of $$\det\Lambda_{bb}$$, which is not a function of $$x_a$$

Together, these two observations imply that $$p(x_a)$$ is itself Gaussian, and thus we can ignore the constant that we get from the integral. Instead, given that the distribution is Gaussian, we can cherry pick $$\mu_a$$ and $$\Sigma_a$$ by comparing coefficients with the general Gaussian expansion
<a id="gaussian"></a>

$$
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) = -\frac{1}{2}x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu + \text{const}.
$$

Alternatively, we could manually expand the integral and compute everything, but comparing coefficients is much easier.

The full expression for $$f(x_a)$$ has terms from our [original expansion](#og-expansion), and the leftover terms from completing the square on $$x_b$$. The terms coming from our original expansion are

$$
-\frac{1}{2}x_a^T\Lambda_{aa}x_a + x_a^T(\Lambda_{aa}\mu_a + \Lambda_{ab}\mu_b),
$$

while the terms leftover from completing the square are

$$
\begin{align*}
&\frac{1}{2}(\Lambda_{bb}\mu_b - \Lambda_{ba}x_a + \Lambda_{ba}\mu_a)^T\Lambda_{bb}^{-1}(\Lambda_{bb}\mu_b - \Lambda_{ba}x_a + \Lambda_{ba}\mu_a)\\
&= \frac{1}{2}(-\mu_b^T\Lambda_{ba}x_a-x_a^T\Lambda_{ba}^T\mu_b + x_a^T\Lambda_{ba}^T\Lambda_{bb}^{-1}\Lambda_{ba}x_a \\
&\qquad - x_a^T\Lambda_{ba}^T\Lambda_{bb}^{-1}\Lambda_{ba}\mu_a-\mu_a^T\Lambda_{ba}^T\Lambda_{bb}^{-1}\Lambda_{ba}x_a) + \text{const.} \\
&= \frac{1}{2}x_a^T\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}x_a + x_a^T(-\Lambda_{ab}\mu_b-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}\mu_a) + \text{const.},
\end{align*}
$$

where we combine like terms using $$\Lambda_{ab}=\Lambda_{ba}^T$$ and the fact that all individual terms are scalars, i.e., we can transpose terms freely. Combining all terms together now gives

$$
-\frac{1}{2}x_a^T(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})x_a + x_a^T(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_a + \text{const.},
$$

so comparing coefficients with the [general gaussian expansion](#gaussian) gives

$$
\text{Cov}[p(x_a)] = (\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1},
$$

and thus

$$
\mathbb{E}[p(x_a)] = \text{Cov}[p(x_a)]^{-1}(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1}\mu_a = \mu_a.
$$

Lastly, we note that $$\text{Cov}[p(x_a)]$$ is actually the inverse [Schur complement](https://en.wikipedia.org/wiki/Schur_complement) of block $$\Sigma_{aa}$$, so the result is more cleanly written $$\text{Cov}[p(x_a)] = \Sigma_{aa}$$.

Note how intuitively nice this result is. It's exactly what we expect, even if it takes some work to prove it rigorously.

<a id="2.2"></a>

#### 2.2. Computing marginal gaussian given other other marginal and conditional

The next question we will focus on is computing the marginal $$p(y)$$ given

$$
p(x) = \mathcal{N}(x; \mu, \Lambda^{-1}) \qquad p(y|x) = \mathcal{N}(y; Ax+b,L^{-1}).
$$

This form seems a little contrived but becomes useful when we discuss [DDIMs](#ddim).

The first observation is that the joint $$p(x,y)$$ is Gaussian, since $$\mathbb{P}[x=x_0,y=y_0] = \mathbb{P}[x=x_0]\cdot \mathbb{P}[y=y_0|x=x_0]$$ for all $$(x_0,y_0)$$. Note that it is not true in general that the product of two Gaussian random variables is Gaussian; it works out here because the pdf of the joint distribution at every point is equal to the product of the pdfs of the marginal Gaussians, and the product of Gaussian pdfs is always Gaussian.

Thus, if we can find mean and variance of $$p(x,y)$$, we can use our results from [2.1](#2.1) to obtain the distribution of the other conditional.

We have

$$
\begin{align*}
\ln p(x,y) &= -\frac{1}{2}(x-\mu)^T\Lambda (x-\mu) - \frac{1}{2}(y-Ax-b)^TL(y-Ax-b) + \text{const} \\
&= -\frac{1}{2}\left(x^T\Lambda x + x^TA^TLAx + y^TLy - y^TLA - x^TA^TLy \right. \\
&\qquad\qquad \left. - 2x^T\Lambda \mu + 2x^TA^TLb - 2y^TLb\right) + \text{const} \\
&= -\frac{1}{2}\begin{pmatrix}x \\ y\end{pmatrix}^T\begin{pmatrix}\Lambda + A^TLA & -A^TL \\ -LA & L\end{pmatrix}\begin{pmatrix}x \\ y\end{pmatrix} + \begin{pmatrix}x \\ y\end{pmatrix}^T\begin{pmatrix}\Lambda\mu - A^TLb \\ Lb\end{pmatrix} + \text{const}.
\end{align*}
$$

Comparing coefficients with the [general gaussian expansion](#gaussian), we have

$$
\text{Cov}(x,y) = \begin{pmatrix}\Lambda + A^TLA & -A^TL \\ -LA & L\end{pmatrix}^{-1} = \begin{pmatrix}\Lambda^{-1} & \Lambda^{-1}A^T \\ A\Lambda^{-1} & L^{-1} + A\Lambda^{-1}A^T\end{pmatrix}.
$$

and thus 

$$
\mathbb{E}(x,y) = \begin{pmatrix}\Lambda^{-1} & \Lambda^{-1}A^T \\ A\Lambda^{-1} & L^{-1} + A\Lambda^{-1}A^T\end{pmatrix}^{-1}\begin{pmatrix}\Lambda\mu -A^TLb \\ Lb\end{pmatrix} = \begin{pmatrix}\mu \\ A\mu + B\end{pmatrix}.
$$

Finally, using [2.1](#2.1),

$$
\mathbb{E}(y) = A\mu + b \qquad \text{Cov}(y) = L^{-1} + A\Lambda^{-1}A^T.
$$

With some more work, we could also extract the other conditional, but we won't need this result.

Like our result in [2.1](#2.1), this final expression is nice because it aligns reasonably with what we expect. We have $$y|x$$ sampled from a distribution with mean $$f(x)$$, where $$f$$ is linear; therefore, the fact that $$\mathbb{E}[y] = f(\mathbb{E}(x))$$ makes intuitive sense. Further, since we have $$f(x) = Ax+b$$, we expect $$\text{Cov}(f(x)) = A^T\text{Cov(x)}A$$. The only "dependence" that $$y$$ has on $$x$$ is through their means; $$\text{Cov(y|x)} = L^{-1}$$ is a source of noise that is essentially independent from the noise associated with $$x$$, so through linearity of variance it makes intuitive sense that $$\text{Cov}(y) = A^T\text{Cov(x)}A + L^{-1}$$.

<a id="2.3"></a>

#### 2.3. KL Divergence

The last thing we will examine is how to compute the KL divergence, or relative entropy, between two multivariate gaussians. The general expression is given by 

$$
D_{KL}(P || Q) = \mathbb{E}_{x\sim p(x)}\left[\log \frac{p(x)}{q(x)}\right].
$$

Applying this to multivariate $$P=\mathcal{N}(\mu_1, \Sigma_1)$$ and $$Q=\mathcal{N}(\mu_2, \Sigma_2)$$, we have

$$
\begin{align*}
\mathbb{E}_{x\sim p(x)}\left[\log \frac{p(x)}{q(x)}\right] &= \frac{1}{2}\mathbb{E}_{x\sim p(x)}\left[\log \frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} - (x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) + (x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)\right].
\end{align*}
$$

To simplify this, we can apply the [trace trick](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#The_trace_of_a_1_.C3.97_1_matrix); since quadratic forms $$x^TAx$$ are singletons, it is equal to its trace, and since traces commute, it is thus also equal to $$\text{Tr}((Ax)x^T)$$ and $$\text{Tr}(x(x^TA))$$. So, we can simplify our expression as follows:

$$
\begin{align*}
&\frac{1}{2}\log\frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} + \frac{1}{2}\mathbb{E}_{x\sim p(x)}\left[\text{Tr}(-\Sigma_1^{-1}(x-\mu_1)(x-\mu_1)^T) + \text{Tr}(\Sigma_2^{-1}(x-\mu_2)(x-\mu_2)^T)\right] \\
&= \frac{1}{2}\log\frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} + \frac{1}{2}\mathbb{E}_{x\sim p(x)}\left[\text{Tr}(-\Sigma_1^{-1}\Sigma_1) + \text{Tr}(\Sigma_2^{-1}(xx^T-2x\mu_2+\mu_2\mu_2^T))\right] \\
&= \frac{1}{2}\log\frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} -\frac{1}{2}d + \frac{1}{2}\mathbb{E}_{x\sim p(x)}\left[\text{Tr}(\Sigma_2^{-1}(x^Tx-2x^T\mu_2+\mu_2^T\mu_2)\right)] \\
&= \frac{1}{2}\log\frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} -\frac{1}{2}d + \frac{1}{2}\mathbb{E}_{x\sim p(x)}\left[\text{Tr}(\Sigma_2^{-1}((\Sigma_1-\mu_1^T\mu_1+2x^T\mu_1)-2x^T\mu_2+\mu_2^T\mu_2)\right)] \\
&= \frac{1}{2}\log\frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} -\frac{1}{2}d + \frac{1}{2}\text{Tr}(\Sigma_2^{-1}\Sigma_1) + \frac{1}{2}\mathbb{E}_{x\sim p(x)}\left[\text{Tr}(\Sigma_2^{-1}(\mu_1-\mu_2)^T(\mu_1-\mu_2)\right)] \\
&= \frac{1}{2}\log\frac{\vert \Sigma_2\vert}{\vert \Sigma_1\vert} -\frac{1}{2}d + \frac{1}{2}\text{Tr}(\Sigma_2^{-1}\Sigma_1) + \frac{1}{2}(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2).
\end{align*}
$$

<a id="ddim"></a>

### 3. DDIM

Next, we turn to DDIMs [(Song et al. 2020)](https://arxiv.org/abs/2010.02502), since this variation on DDPMs is an important component of the Stable Diffusion models. A key motivation for these models is the fact referenced above that inference, i.e., simulating the backwards diffusion process, is quite expensive.

This paper introduces a family of inference distributions $$\mathcal{Q}$$ paramaterized by $$\sigma$$ fixing the variance added during denoising:

$$
q_{\sigma}(x_{t-1}|x_t,x_0) = \mathcal{N}\left(\sqrt{\overline{\alpha}_{t-1}}x_0 + \sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}\cdot \frac{x_t - \sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}}, \sigma_t^2 I\right).
$$

This distribution was constructed so that forward sampling still works as expected for the purpose of training; it can be proven that $$q_{\sigma}(x_t|x_0) = \mathcal{N}(\sqrt{\overline{\alpha}_t}x_0, (1-\overline{\alpha}_t)I)$$, which is the same sampling distribution for DDPMs (see [here](#forward-sampling)). It is not true that the normal forwards process $$q(x_t|x_{t-1})$$ stays intact, and in fact $$q_{\sigma}(x_t|x_{t-1},x_0)\neq q_{\sigma}(x_t|x_{t-1})$$, hence this family of distributions is _non-Markovian_.

To see why this new family of distributions intuitively captures the spirit of the backwards diffusion process derived in DDPMs, note that

$$
\frac{x_t - \sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}} = \varepsilon_{t},
$$

so the two terms $$\sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}$$ (under the mean) and $$\sigma_t$$ (under actual variance) can be seen as having total noise $$\sqrt{1-\overline{\alpha}_{t-1}}$$, which matches the noise expression for the forward process, i.e., the distribution $$q(x_{t-1}|x_0)$$. The paper introduces this "splitting" of the noise factors to control the actual amount of noise that is induced during the backwards inference step.

#### 3.1. Proof that $$\mathcal{Q}$$ satisfies forwards definition

To prove that forwards sampling from $$x_0$$ remains the same, we can use an inductive argument, inducting downwards on the timestep. When $$T=t$$, we assume that $$q_{\sigma}(x_T|x_0)$$ is normally distributed (i.e., pure noise), so

$$
q_{\sigma}(x_T|x_0) = \mathcal{N}(\sqrt{\overline{\alpha}_T}x_0, (1-\overline{\alpha}_T)I) = \mathcal{N}(0,I),
$$

and our base case holds. Now, by our inductive hypothesis, assume that we have

$$
q_{\sigma}(x_t|x_0) = \mathcal{N}(\sqrt{\overline{\alpha}_t}x_0, (1-\overline{\alpha}_t)I),
$$

and we also have

$$
q_{\sigma}(x_{t-1}|x_t,x_0) = \mathcal{N}(\sqrt{\overline{\alpha}_{t-1}}x_0 + \sqrt{1 - \overline{\alpha}_{t-1} - \sigma_t^2}\cdot \frac{x_t - \sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}}, \sigma_t^2 I).
$$

Now we have a marginal distribution $$q_{\sigma}(x_{t}|x_0)$$ and a distribution conditioned on this marginal distribution $$q_{\sigma}(x_{t-1} | x_t,x_0)$$. We wish to find the other marginal $$q_{\sigma}(x_{t-1}|x_0)$$, and luckily the setup is the same as our setup from [2.2](#2.2). Thus, using the results from our derivation, we have

$$
q_{\sigma}(x_{t-1}|x_0) = \mathcal{N}(A\mu + b, L^{-1} + A\Lambda^{-1}A^T),
$$

where

$$
A\mu + b = \sqrt{\overline{\alpha}_{t-1}}x_0 + \sqrt{1 - \overline{\alpha}_{t-1}-\sigma_t^2}\cdot \frac{\sqrt{\overline{\alpha}_t}x_0 - \sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}} = \sqrt{\overline{\alpha}_{t-1}}x_0,
$$

and

$$
L^{-1} + A\Lambda^{-1}A^T = \sigma_t^2I + \frac{1 - \overline{\alpha}_{t-1} - \sigma_t^2}{1 - \overline{\alpha}_t}(1-\overline{\alpha}_t)I = (1-\overline{\alpha}_{t-1})I.
$$

Thus,

$$
q_{\sigma}(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\overline{\alpha}_{t-1}}x_0, (1-\overline{\alpha}_{t-1})I),
$$

which completes the proof.

#### 3.2. Proof that $$\mathcal{Q}$$ can be applied to DDPM trained models

One of the key properties that the inference distributions $$\mathcal{Q}$$ is that they can be applied even to models that were originally trained with the DDPM objective.

In order to prove this, we first introduce some more notation from the paper. Let $$\mathcal{L}$$ be a family of loss functions generalizing the DDPM training process, such that for any $$L_{\gamma}\in \mathcal{L}$$,

$$
L_{\gamma}(\varepsilon_t) = \mathbb{E}_{x_0, \varepsilon_t, x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\varepsilon_t}\left[\gamma \lVert\varepsilon_t - \varepsilon_{\theta}(x_t)\rVert_2^2\right].
$$

For example, in DDPM, it was shown that the mathematically optimal loss was given by

$$
\gamma = \frac{(1-\alpha_t)^2}{2\alpha_t(1-\overline{\alpha}_t)\lVert \Sigma_{\theta}(x_t,t)\rVert^2},
$$

(see [here](#ddpm-math-loss)), while $$\gamma = 1$$ was shown to be good for training. Now, let $$J_{\sigma}$$ be the optimal objective for learning $$q_{\sigma}$$. To show that $$\mathcal{Q}$$ inference can be effectively applied to DDPM-trained models, it suffices to show that $$J_{\sigma}\in \mathcal{L}$$.

We first utilize the variational inference objective from [(Ho et al. 2020)](https://arxiv.org/abs/2006.11239):

$$
L_{VLB} = \mathbb{E}_{x_{0:T}\sim q_{\sigma}(x_{0:T})}\left[\frac{\log q_{\sigma}(x_{1:T\vert x_0})}{\log p_{\theta}(x_{0:T})}\right]
$$

Using results from [(Sohl-Dickstein et al. 2015)](https://arxiv.org/abs/1503.03585) and the same derivations from DDPM, we have

$$
J_{\sigma}(\varepsilon_{\theta}) \equiv \mathbb{E}_{x_{0:T}\sim q_{\sigma}(x_{0:T})}\left[\sum_{t=2}^{T}D_{KL}((q_{\sigma}(x_{t-1}\vert x_t, x_0))\Vert p_{\theta}^{(t)}(x_{t-1}\vert x_t)) - \log p_{\theta}^{(1)}(x_0 \vert x_1)\right]
$$

when only taking terms $$L_1,\dots,L_{t-1}$$ (in the notation of the paper, we use $$\equiv$$ instead of $$=$$ when we take steps that throw away constant factors).

Now, per the paper, we define the actual generative process $$p_{\theta}(x_{0:T})$$ as a function of the derived distribution $$q_{\sigma}(x_{t-1}\vert x_t, x_0)$$. Since we don't know $$x_0$$ during inference, we replace this term in $$q_{\sigma}$$ with the (derived) output of the neural net.

From the definition of the forward process (see [here](#ddpm)), our predicted denoised observation of $$x_0$$ given $$x_t$$ is given by

$$
f_{\theta}^{(t)}(x_t) := \frac{(x_t - \sqrt{1-\alpha_t} \cdot \varepsilon_{\theta}^{(t)}(x_t))}{\sqrt{\alpha_t}}.
$$

Thus, we can define the reverse generative process with a prior distribution $$p_{\theta}^{(t)}(x_{t}) = \mathcal{N}(0, I)$$ as

$$
p_{\theta}^{(t)}(x_{t-1} \vert x_{t}) = q_{\sigma}(x_{t-1} \vert x_t, f_{\theta}^{(t)}(x_t)).
$$

Finally, we have enough to evaluate $$J_{\sigma}$$. First note that we can rewrite the expected value:

$$
J_{\sigma}(\varepsilon_{\theta}) \equiv \mathbb{E}_{x_0,\varepsilon,x_t=\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\varepsilon}[D_{KL}(q_{\sigma}(x_{t-1}|x_t,x_0))||q_{\sigma}(x_{t-1} \vert x_t, f_{\theta}^{(t)}(x_t))].
$$

Using [2.3](#2.3), we see that this is equivalent to optimizing mse between the two means (under $$\equiv$$), so the part of our objective inside of the expected value becomes

$$
\begin{align*}
& \mathbb{E}_{x_0,\varepsilon,x_t=\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\varepsilon}\left[
  \left\lVert\left(\sqrt{\overline{\alpha}_{t-1}}x_0 + \sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}\cdot \frac{x_t - \sqrt{\overline{\alpha}_t}x_0}{\sqrt{1-\overline{\alpha}_t}}\right)\right.\right. \\
&\qquad\qquad
  \left.\left. -\left(\sqrt{\overline{\alpha}_{t-1}}f_{\theta}^{(t)}(x_t) + \sqrt{1-\overline{\alpha}_{t-1}-\sigma_t^2}\cdot \frac{x_t - \sqrt{\overline{\alpha}_t}f_{\theta}^{(t)}(x_t)}{\sqrt{1-\overline{\alpha}_t}}\right)\right\rVert^2\right].
\end{align*}
$$

Further simplifying,

$$
\begin{align*}
J_{\sigma}(\varepsilon_{\theta}) &\equiv \mathbb{E}_{x_0,\varepsilon,x_t=\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\varepsilon}\left[\lVert x_0 - f_{\theta}^{(t)}(x_t)\rVert^2\right] \\
&\equiv \mathbb{E}_{x_0,\varepsilon,x_t=\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\varepsilon}\left[\left\lVert \frac{x_t - \sqrt{1-\overline{\alpha}_t}\varepsilon}{\sqrt{\overline{\alpha}_t}} - \frac{x_t - \sqrt{1-\overline{\alpha}_t}\varepsilon_{\theta}^{(t)}(x_t)}{\sqrt{\overline{\alpha}_t}} \right\rVert^2\right]\\
&\equiv \mathbb{E}_{x_0,\varepsilon,x_t=\sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\varepsilon}[\lVert \varepsilon - \varepsilon_{\theta}^{(t)}(x_t)\rVert^2] \in \mathcal{L},
\end{align*}
$$

as desired.

### 4. Conditional generation

Eventually, our goal is not just to generate image from noise, but we would like to also generate images conditioned on text labels. More specifically, our final task is to eventually generate images conditioned on both text labels _and_ an input reference image, but we'll discuss this more in the [next section](#final-conditional-objective). In this section, we'll discuss methods for conditional generation in an easier subtask, which starts with generating images from discrete input classes. As a simple example of this subtask, we might have a diffusion model try to generate one of the ten digits from the MNIST dataset.

#### 4.1 Classifier Guided

The general approach is as follows. At each step during inference, we are trying to approximate $$\nabla_{x_t}\log q(x_t)$$. 

We first consider the case when we have an external classifier $$f_{\phi}(y|x_t,t)$$.

#### 4.2 Classifier-free

<a id="final-conditional-objective"></a>

### 5. Modifying the conditional objective for Pix2Pix

## References

[1] Brooks et al. ["InstructPix2Pix: Learning to Follow Image Editing Instructions"](https://arxiv.org/abs/2211.09800) (2022).

[2] Rombach et al. ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752) (2021).

[3] Brown et al. ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (2020).

[4] Ho et al. ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) (2020).

[5] Song et al. ["Denoising Diffusion Implicit Models"](https://arxiv.org/abs/2010.02502) (2020).

[6] Weng, Lilian. ["What are diffusion models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) Lil’Log (2021).

[7] Bishop, Christopher ["Pattern Recognition and Machine Learning"](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006).

[8] Sohl-Dickstein et al. ["Deep unsupervised learning using nonequilibrium thermodynamics."](https://arxiv.org/abs/1503.03585) (2015).
