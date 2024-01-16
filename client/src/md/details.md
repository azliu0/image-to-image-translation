---
title: About this website
author: Jack Chen and Andrew Liu
date: December 25, 2023
time: 5 min
---

This website is a visualizer and custom implementation of InstructPix2Pix ([Brooks et al. 2022](https://arxiv.org/abs/2211.09800)), an image-to-image diffusion model that incorporates text as conditional guidance to the input. The base architecture is Stable Diffusion ([Rombach et al. 2021](https://arxiv.org/abs/2112.10752)) with some additional inference heuristics  inspired by Classifier-Free Guidance ([Ho et al. 2022](https://arxiv.org/abs/2207.12598)).

## 0. Credit

Our implementation of the model is inspired heavily by [Umar Jamil](https://github.com/hkproj/pytorch-stable-diffusion). We found the corresponding YouTube video to be very helpful, and most key components of our pipeline follow the same implementation as his tutorial!

## 1. How to use this website 

### Models

The general capabilities of the model include being able to transform an input image, some input text, and generate a new image that is guided on both inputs. We plan to support eventually support 6 different variations of this model <span style="color: gray;">(models in gray are not supported yet)</span>:

- **pix2pix-base**: the original paper model as it appears on [HuggingFace](https://huggingface.co/docs/diffusers/training/instructpix2pix)

- **pix2pix-full-no-cfg-no-ddim**: our version of the model with all major architectural components; this is essentially Stable Diffusion. this version of the model does not use Classifier Free Guidance (CFG) or DDIM as inference heuristics. By "CFG", we refer to the slightly refined version of CFG that was introduced in InstructPix2Pix, to specifically support the extra class conditional input (reference image).

<div style="color: gray;">

- pix2pix-full: our version of the model with all major architectural components, and with CFG and DDIM as inference heuristics

- pix2pix-full-no-ddim: our version of the model, with CFG but no DDIM for inference

- pix2pix-1: our diffusion model, with the variational autoencoder and transformer taken from a pre-trained library. no CFG/DDIM

- pix2pix-2: our diffusion model and variational autoencoder, with the transformer taken from a pre-trained library. no CFG/DDIM
</div>

### Image resolution

Images with 512x512 base resolution will generally work best, since this is the default resolution that is supported by our model. Images that are other sizes will be automatically compressed or expanded into a 512x512 output image. To see some examples of simple images and prompts that seemed to produce decent results, check out the [gallery](/gallery).

### Other parameters

- **Negative Prompt**: in contrast with the generation input text, any text in the negative prompt will guide the image *away* from its context. 

- **Inference steps** (min: 2, max: 1000, int): the number of scheduler (ddpm/ddim) steps to run the backwards diffusion process on. in general, a higher number of steps leads to higher quality images, but will take much longer to generate. 

- **Temperature** (min: 0.1, max: 1, float): temperature represents the strength of the diffusion model, compared to the reference image. a temperature of 1 indicates that the diffusion model is acting independently of the input image, while lower temperatures will produce results closer to the reference image.

- **CFG** (min: 1, max: 14, int): this parameter is only available for models that use Classifier-Free Guidance. higher values indicate that the score for the contextualized inputs should be weighted higher. In other words, a high CFG scale gives more power to the captions (both positive and negative), while lower values will give more power to the diffusion model.

## 3. Website details

This website was built from scratch. The main frontend framework was [React](https://react.dev/), and we had some fun integrating libraries like [Mantine UI](https://mantine.dev/) and [Framer Motion](https://www.framer.com/motion/) for styling quirks. We used [react-markdown](https://github.com/remarkjs/react-markdown) for markdown parsing, along with [remark-math](https://www.npmjs.com/package/remark-math) and [react-katex](https://www.npmjs.com/package/react-katex) to handling $$\LaTeX$$ parsing inside of markdown.

The backend is [Flask](https://flask.palletsprojects.com/en/3.0.x/). For the purposes of keeping things simple, we decided not to deploy our model to any endpoints and instead all of the inference is done on the deployment server. If we had the funds for a GPU on the server, this would be OK; unfortunately, we do not, so inference is quite slow. For the time being, we don't have plans to fix this issue, since it seems like anything that costs money will end up costing too much to keep this website running for a very long period of time.

We used [Render](https://render.com/) for deployment. Feel free to check out our [repository](https://github.com/azliu0/image-to-image-translation)!

## 4. Conclusion

This was a very fulfulling winter break project for us, since we are both very interested in generative modelling and excited about its potential. We are glad to have gained some experience understanding these models at a deeper level. We also found the creation of this website to be a good exercise of our web development skills.