---
title: About this website
author: Jack Chen and Andrew Liu
date: December 25, 2023
time: 10 min
---

[under construction]

This website is a visualizer for InstructPix2Pix ([Brooks et al. 2022](https://arxiv.org/abs/2211.09800)), an image-to-image diffusion model that incorporates text as conditional guidance to the input. The architecture uses a combination of text-to-image Stable Diffusion ([Rombach et al. 2021](https://arxiv.org/abs/2112.10752)) with OpenAI's text transformer GPT-3 ([Brown et al. 2020](https://arxiv.org/abs/2005.14165)).

The general capabilities of the model include being able to transform an input image, some input text, and generate a new image that is guided on both inputs. 

[perhaps insert a sample image here?]

We train a few models from scratch that recreates the architecture in the original InstructPix2Pix paper, where the results of the models can be played around with in this website. Uploaded images should be square; if not, they are automatically cropped. On the backend, all images are transformed to 256x256 resolution, so images with 256x256 base resolution will generally work best. To seem some examples of simple images and prompts that seemed to produce decent results, check out the [gallery](/gallery). 

## Model details

[under construction]

Eventually, we will publish some details here about what models we coded and trained for this website.

Model details...

Dataset details...

Training details...

## Summary of math

Coming into this project, we wanted to have a solid understanding of the math underlying the model. We reproduce some of the things we learned here. Since we had some background knowledge, we'll only rigorously rederive results that we learned or found very interesting.

### 1. DDPM Speedrun

We begin with DDPMs [(Ho et al. 2020)](https://arxiv.org/abs/2006.11239). In these diffusion models, we noise images sampled from distribution $q(x)$ through the *forward process*, defined via

$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})\qquad q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I),
$$

where $$\beta_t$$ is some noise scheduler. We'll typically see $$\beta_1 < \beta_2 < \cdots < \beta_T$$ to ensure that the final image is pure noise.

Let $$\alpha_t = 1-\beta_t$$ and $$\prod_{i=1}^t \alpha_t = \overline{\alpha}_t$$. It can be shown that 

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\overline{\alpha}_t}x_0, (1-\overline{\alpha}_t)I),
$$

which not only makes sampling the forward process very efficient, but also shows that we can think of the forward process as some linear combination of the original image and pure noise.

In the *backward process*, the goal is to produce a model $$p_{\theta}$$ that approximates $$q$$, where we define 

$$
p_{\theta}(x_{0:T}) = p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t) \qquad p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t)).
$$

Given perfect $$p_{\theta}$$, we can perfectly recreate the original data distribution from pure noise, which is the magic of diffusion. It turns out that $$q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}(x_t,x_0), \tilde{\beta}_tI)$$ is tractable, and it can be shown that 

$$
\tilde{\mu}(x_t,x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\varepsilon_t\right),
$$

where $$\varepsilon_t$$ is the noise added to produce $$x_t$$ from $$x_0$$. This quantity is relevant because it can be shown that optimizing the [variational lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) is equivalent to minimizing

$$
L = \mathbb{E}_{x_0, \varepsilon, t}[k\lVert \tilde{\mu}_t(x_t,x_0) - \mu_{\theta}(x_t,t)\rVert^2].
$$

where $$\mu_{\theta}(x_t,t)$$ is the backwards mean predicted by our model, and $$k$$ is a constant that depends on some important factors. [(Ho et al. 2020)](https://arxiv.org/abs/2006.11239) showed that this constant is not super important, so eventually reduced the loss function to 

$$
L = \mathbb{E}_{x_0, \varepsilon, t}[\lVert \varepsilon_t - \varepsilon_{\theta}(x_t,t)\rVert^2].
$$

The full derivation for this fact is recreated [here](https://azliu0.github.io/mit-notes/6.7900/6_7900_Notes.pdf). The importance of this fact is that training DDPM models boils down to sampling random images at random timesteps, and having a model (typically a UNet) learn to predict the the noise added at timestep $$t$$. 

Some important training details that we found confusing: 
1. the input of our UNet is an image $$x_t$$, generated from $$x_0$$ using the distribution $$q(x_t|x_0)$$. the output of our UNet is the noise used to generate $$x_t$$, $$\varepsilon_{\theta}(x_t,t)$$.

2. this noise is sampled from $$\mathcal{N}(0,1)$$, per the [reparamaterization trick](https://en.wikipedia.org/wiki/Variational_autoencoder#Reparameterization). this noise represents normalized quantity of noise added from $$x_0$$ to $$x_t$$, **not** from $$x_{t-1}$$ to $$x_t$$. in other words, this noise corresponds to the forwards distribution $$q(x_t|x_0)$$, and not $$q(x_t|x_{t-1})$$.

3. even though we are predicting noise from $$x_0$$ to $$x_t$$, this noise is used to generate the distribution $$p(x_{t-1}|x_t)$$. in other words, the distribution for predicting one timestep backwards from $$t$$ to $$t-1$$ is a function of noise added from $$0$$ to $$t$$. this is because the original (tractable) distribution that we are trying to learn is $$q(x_{t-1}|x_t,x_0)$$, which also has access to information about $$x_0$$. 

4. a concrete demonstration of this fact is the way that we simulate the backwards process once we have trained our UNet. We must make $$T$$ calls to the UNet to go from pure noise to the original data distribution (more accurately, our approximation of the distribution). In each of these $$T$$ calls, although the UNet estimates noise from timestep $$0$$ to the current timestep, we cannot jump directly to the beginning, because we do not know $$p(x_0|x_t)$$ as a function of this noise. thus, although we have a shortcut for the forwards process, there is no equivalent shortcut for the backwards process, and so inference is quite expensive.

### DDIM

This training 

## Website details

[under construction]

This website was built from scratch. The main frontend framework was [React](https://react.dev/), and we had some fun integrating libraries like [Mantine UI](https://mantine.dev/) and [Framer Motion](https://www.framer.com/motion/) for styling quirks. We used [react-markdown](https://github.com/remarkjs/react-markdown) for markdown parsing.

[backend??]. We faced some challenges deploying the models. 

Feel free to check out our [repository](https://github.com/azliu0/image-to-image-translation). 

## Conclusion

[under construction]

This was a very fulfulling winter break project for us, since we are both very interested in generative modelling and excited about its potential. We are glad to have gained some experience understanding these models at a deeper level. We also found the creation of this website to be a good exercise of our web development skills.

[how good are the models?...]

## References

[1] Brooks et al. ["InstructPix2Pix: Learning to Follow Image Editing Instructions"](https://arxiv.org/abs/2211.09800) (2022). 

[2] Rombach et al. ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752) (2021).

[3] Brown et al. ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (2020).

[4] Ho et al. ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) (2020).

[5] Song et al. ["Denoising Diffusion Implicit Models"](https://arxiv.org/abs/2010.02502) (2020).

[6] Weng, Lilian. ["What are diffusion models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) Lil’Log (2021).

