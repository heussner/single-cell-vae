import os
from cv2 import normalize
from matplotlib import pyplot as plt
import torch
from torch import optim
from models.base import BaseVAE
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from ml_collections import ConfigDict


class SingleCellVAE(pl.LightningModule):
    def __init__(
        self,
        model: BaseVAE,
        optim_config: ConfigDict,
        n_samples: int,
        sample_step: int = 10,
    ) -> None:
        super(SingleCellVAE, self).__init__()
        self.model = model
        self.optim = optim_config
        self.n_samples = n_samples
        self.sample_step = sample_step
        self.step = 0
        self.colors = ["Reds", "Greens", "Blues", "Purples", "Oranges"]

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img = batch
        self.curr_device = real_img.device

        if not hasattr(self, "save_dir"):
            self.__setattr__(
                "save_dir",
                f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/imgs/",
            )
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

        results = self.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            M_N=batch.size(0) / self.n_samples,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        for k, v in train_loss.items():
            self.logger.experiment.add_scalar(k, v, self.step)

        if batch_idx % self.sample_step == 0:
            recons = results[0]
            inputs = results[1]
            save_path = self.save_dir + f"{self.step}.png"
            for k in range(inputs.size(1)): 
                comp = torch.cat(
                    (
                        inputs[:64, k, :, :].unsqueeze(1), #from 30:-30, 30:-30, to :,:
                        recons[:64, k, :, :].unsqueeze(1)
                    ), 
                    -1)
                grid = make_grid(comp, normalize=True)
                assert (
                    (grid[0, :, :] == grid[1, :, :]).all().item() and 
                    (grid[0, :, :] == grid[2, :, :]).all().item()
                )
                cmap = plt.cm.get_cmap(self.colors[k]).copy()
                cmap.set_bad(color="black")
                plt.imsave(
                    os.path.join(self.save_dir, 
                    f"step_{self.step}_c{k}.png"), 
                    grid[0,:,:].detach().cpu().numpy(), 
                    cmap=cmap
                )
                self.logger.experiment.add_image(
                    "Inputs-Reconstructions", grid[:1,:,:], self.step
                )

            #msg = f"loss: {train_loss['loss']:.4f} -- rec: {train_loss['Reconstruction_Loss']:.4f} -- kld (scaled): {train_loss['KLD_Scaled']:.4f} -- kld: {train_loss['KLD']:.4f}"
            #print(msg)

        for k, v in train_loss.items():
            if k != "loss" and v.grad_fn:
                train_loss[k] = v.detach()

        self.step += 1

        self.log_dict(train_loss)

        return train_loss

    def configure_optimizers(self):

        optimizers = {"adam": optim.Adam, "sgd": optim.SGD}
        schedulers = {}

        optims = []
        scheds = []

        optimizer = optimizers[self.optim.type](
            self.model.parameters(),
            lr=self.optim.lr,
            weight_decay=self.optim.weight_decay,
        )

        optims.append(optimizer)

        if not self.optim.scheduler is None:
            scheduler = schedulers[self.optim.scheduler](**self.optim.scheduler_params)
            scheds.append(scheduler)

        return optims, scheds
