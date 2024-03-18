from absl import app, flags
from ml_collections.config_flags import config_flags
from models import vae_models
from data import load_data
import numpy as np
import torch
import os
import umap
import pandas as pd
from sklearn.manifold import TSNE
import sys
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("train_config")
config_flags.DEFINE_config_file("model_config")


def main(_):

    config = FLAGS.train_config
    model_config = FLAGS.model_config

    if not config.eval_mode:
        print("ERROR: indicate preference for eval mode in config file.")
        sys.exit()

    if os.path.isdir(config.results_dir):
        print("ERROR: results dir already exists")
        sys.exit()

    model_class = vae_models[model_config.model]
    model = model_class(**model_config.model_params)
    print(f"Loading model from checkpoint: {config.eval_model_path}")
    state_dict = torch.load(config.eval_model_path, map_location="cpu")["state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.split("model.")[-1]] = v
    model.load_state_dict(new_sd)
    model.to(config.device)

    dataset, loader = load_data(config.datapath, eval=True, **config.data_params)
    
    colors = ["Blues", "Greens", "Reds", "Purples", "Oranges"]

    with torch.no_grad():

        files = []
        encoded = []

        save_path = os.path.join(config.results_dir, "imgs")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)        
        for i, (filepaths, img) in enumerate(loader):
            img = img.to(config.device)
            files += filepaths
            recon, inp, mu, log_var = model(img)

            if (i+1) % 2 == 0: #change from 10 to 2
                for k in range(inp.size(1)): 
                    comp = torch.cat(
                        (
                            inp[:64, k, 30:-30, 30:-30].unsqueeze(1), 
                            recon[:64, k, 30:-30, 30:-30].unsqueeze(1)
                        ), 
                        -1)
                    grid = make_grid(comp, normalize=True)
                    assert (
                        (grid[0, :, :] == grid[1, :, :]).all().item() and 
                        (grid[0, :, :] == grid[2, :, :]).all().item()
                    )
                    cmap = plt.cm.get_cmap(colors[k]).copy()
                    cmap.set_bad(color="black")
                    plt.imsave(
                        os.path.join(save_path, str(i) + f"__c{k}.png"), 
                        grid[0,:,:].detach().cpu().numpy(), 
                        cmap=cmap
                    )

            mu = mu.detach().cpu().numpy() #changing to z
            encoded += list(mu)

    tsne = TSNE(n_components=2).fit_transform(np.asarray(encoded))
    ump = umap.UMAP().fit_transform(encoded)

    vae_df = pd.DataFrame(list(zip(files, encoded)), columns=["files", "encode"])
    tsne_df = pd.DataFrame(
        list(zip(files, tsne[:, 0], tsne[:, 1])), columns=["files", "x", "y"]
    )
    umap_df = pd.DataFrame(
        list(zip(files, ump[:, 0], ump[:, 1])), columns=["files", "x", "y"]
    )

    vae_df.to_csv(os.path.join(config.results_dir, config.vae_embed_file), index=False)
    tsne_df.to_csv(
        os.path.join(config.results_dir, config.tsne_embed_file), index=False
    )
    umap_df.to_csv(
        os.path.join(config.results_dir, config.umap_embed_file), index=False
    )


if __name__ == "__main__":
    app.run(main)
