---
title: About
author: Jack Chen and Andrew Liu
date: December 25, 2023
time: 5 min
---

This website is a quick visualizer of InstructPix2Pix ([Brooks et al. 2022](https://arxiv.org/abs/2211.09800)) deployed on [Modal Labs](https://modal.com/). InstructPix2Pix is an image-to-image diffusion model that incorporates text as conditional guidance to the input. The base architecture is Stable Diffusion ([Rombach et al. 2021](https://arxiv.org/abs/2112.10752)) with some additional inference heuristics  inspired by Classifier-Free Guidance ([Ho et al. 2022](https://arxiv.org/abs/2207.12598)).

## Models

Right now the app serves two models:

1. `image-to-image` from the `diffusers` library, which is just imported and run directly.
2. An implementation of `instruct-pix2pix`, which lives in `server/pix2pix/modules`. This implementation was mostly taken from [here](https://github.com/hkproj/pytorch-stable-diffusion).

## Details

### Models

The general capabilities of the model include being able to transform an input image, some input text, and generate a new image that is guided on both inputs. We plan to support eventually support 6 different variations of this model <span style="color: gray;">(models in gray are not supported yet)</span>:

- **pix2pix-base**: the original paper model as it appears on [HuggingFace](https://huggingface.co/docs/diffusers/training/instructpix2pix)

- **pix2pix-full-no-cfg-no-ddim**: the custom implementation of the model.

### Parameters

- **Negative Prompt**: in contrast with the generation input text, any text in the negative prompt will guide the image *away* from its context. 

- **Inference steps** (min: 2, max: 1000, int): the number of scheduler (ddpm/ddim) steps to run the backwards diffusion process on. in general, a higher number of steps leads to higher quality images, but will take much longer to generate. 

- **Temperature** (min: 0.1, max: 1, float): temperature represents the strength of the diffusion model, compared to the reference image. a temperature of 1 indicates that the diffusion model is acting independently of the input image, while lower temperatures will produce results closer to the reference image.

- **CFG** (min: 1, max: 14, int): this parameter is only available for models that use Classifier-Free Guidance. higher values indicate that the score for the contextualized inputs should be weighted higher. In other words, a high CFG scale gives more power to the captions (both positive and negative), while lower values will give more power to the diffusion model.
