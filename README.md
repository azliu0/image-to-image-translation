# image-to-image-translation

## Setup

### Client

```sh
cd client
npm install && npm run dev
```

### Server

After installing dependencies, run:

```sh
python3 wsgi.py
```

## Model

Right now the app serves two models:

1. `image-to-image` from the `diffusers` library, which is just imported and run directly.
2. An implementation of `instruct-pix2pix`, which lives in `server/pix2pix/modules`. This implementation was mostly taken from [here](https://github.com/hkproj/pytorch-stable-diffusion).

The models are deployed on [Modal Labs](https://modal.com/) via [web endpoints](https://modal.com/docs/guide/web-endpoints). 

To download weights, run `cd server/pix2pix && python3 load-weights.py`, which stores them in a [Modal volume](https://modal.com/docs/guide/volumes).

To run inference locally, set `USE_REMOTE` to `False` in `server/config.py`. In this case, you'll need to download the weights locally into the `data/` directory. See `server/pix2pix/load-weights.py` for the relevant huggingface URL.
