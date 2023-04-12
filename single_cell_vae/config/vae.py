import ml_collections


def get_config():
    cfg = ml_collections.ConfigDict()

    cfg.model = "vae"
    cfg.model_params = {
        "img_size": 128,
        "in_channels": 3,
        "latent_dim": 64,
        "hidden_dims": [32, 64, 128, 256, 512],
        "likelihood_dist": "gauss",  # Decoder modeling a 'gauss'ian or 'bern'ouli distribution
    }

    return cfg
