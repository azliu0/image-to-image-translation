from server.pix2pix.modules.clip.clip import CLIP
from server.pix2pix.modules.vae.encoder import VAE_Encoder
from server.pix2pix.modules.vae.decoder import VAE_Decoder
from server.pix2pix.modules.diffusion.unet import Diffusion

import server.pix2pix.modules.model_converter as model_converter


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
