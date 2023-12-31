---
title: How to use this website
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

## Review of literature

Coming into this project, we wanted to have a solid understanding of the math underlying the model. We reproduce some of the things we learned here. 

## Website details

[under construction]

Eventually, we will publish some details here about the tech stack we used to create this website. We faced some challenges deploying the models. In addition to the challenges we faced coding and training the models, we also encountered some unique challenges related to the development and deployment of this site. For now, feel free to check out our [repository](https://github.com/azliu0/image-to-image-translation). 

## Conclusion

[under construction]

We are both very interesting in generative modelling and excited about its potential impact for the world. We are grateful to have gained some experience understanding these models at a deeper level, in addition to the exercise in web development in creating this website. 

## References