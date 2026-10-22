import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)

from lightning.pytorch.loggers import (
    CSVLogger,
)

from forest_disturbance.build import (
    build_datamodule,
    build_mapping,
)

from forest_disturbance.config import (
    load_config,
    save_config,
)

from forest_disturbance.models.lit_module import (
    DisturbanceModule,
)

from forest_disturbance.models.loss import (
    RecentDisturbanceLoss,
)

from experiments.tristan_ltae_late_fusion.model import (
    DualLTAEFusion,
)


class DualAuxDisturbanceModule(
    DisturbanceModule
):
    def __init__(
        self,
        *args,
        aux_weight=0.25,
        eta_min=3e-6,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.aux_weight = (
            aux_weight
        )

        self.eta_min = eta_min

    def _shared_step(
        self,
        batch,
        stage,
    ):
        # Validation uses the exact
        # original evaluation pipeline.
        if stage != "train":
            return super()._shared_step(
                batch,
                stage,
            )

        (
            logits,
            s2_logits,
            s1_logits,
        ) = self.model(
            batch,
            return_aux=True,
        )

        main_loss = self.loss(
            logits,
            batch["target"],
        )

        s2_loss = self.loss(
            s2_logits,
            batch["target"],
        )

        s1_loss = self.loss(
            s1_logits,
            batch["target"],
        )

        loss = (
            main_loss
            + self.aux_weight
            * s2_loss
            + self.aux_weight
            * s1_loss
        )

        batch_size = len(
            batch["sample_id"]
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.log(
            "train/main_loss",
            main_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train/s2_aux_loss",
            s2_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train/s1_aux_loss",
            s1_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return logits, loss

    def configure_optimizers(
        self,
    ):
        optimizer = (
            torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=(
                    self.weight_decay
                ),
            )
        )

        max_epochs = max(
            1,
            int(
                self.trainer.max_epochs
            ),
        )

        if max_epochs == 1:
            return optimizer

        # Epoch 1 warm-up.
        warmup = (
            torch.optim.lr_scheduler
            .LinearLR(
                optimizer,
                start_factor=0.10,
                end_factor=1.0,
                total_iters=1,
            )
        )

        # Then cosine decay.
        cosine = (
            torch.optim.lr_scheduler
            .CosineAnnealingLR(
                optimizer,
                T_max=max(
                    1,
                    max_epochs - 1,
                ),
                eta_min=self.eta_min,
            )
        )

        scheduler = (
            torch.optim.lr_scheduler
            .SequentialLR(
                optimizer,
                schedulers=[
                    warmup,
                    cosine,
                ],
                milestones=[1],
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "overrides",
        nargs="*",
    )

    args = parser.parse_args()

    config = load_config(
        "configs/statistical_mlp.yaml",
        args.overrides,
    )

    config["model"] = {
        "name": "dual_ltae_s1s2",

        "s2_dim": 128,
        "s1_dim": 96,

        "s2_heads": 8,
        "s1_heads": 4,

        "fusion_dim": 512,
        "dropout": 0.10,

        "s2_acquisition_dropout": 0.20,
        "s1_acquisition_dropout": 0.10,

        "aux_weight_each": 0.25,
    }

    output_dir = (
        Path(
            config["output_root"]
        )
        / (
            f"{config['run_name']}_"
            f"{datetime.now():%Y%m%d-%H%M%S}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config(
        config,
        output_dir
        / "config.yaml",
    )

    print(
        "Run folder:",
        output_dir,
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    L.seed_everything(
        config["seed"],
        workers=True,
    )

    mapping = build_mapping(
        config
    )

    datamodule = (
        build_datamodule(
            config,
            mapping,
        )
    )

    model = DualLTAEFusion(
        num_classes=(
            mapping.num_classes
        ),
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        "Foundation-model "
        "parameters: 0"
    )

    loss = RecentDisturbanceLoss(
        no_disturbance_id=(
            mapping.no_disturbance_id
        ),
        **config["loss"],
    )

    module = (
        DualAuxDisturbanceModule(
            model,
            loss,
            mapping,

            **config["optimizer"],

            output_dir=(
                output_dir
            ),

            evaluation=(
                config["evaluation"]
            ),

            aux_weight=0.25,
            eta_min=3e-6,
        )
    )

    tc = config["trainer"]

    trainer = L.Trainer(
        accelerator=(
            tc["accelerator"]
        ),

        devices=tc["devices"],

        precision=(
            tc["precision"]
        ),

        max_epochs=(
            tc["max_epochs"]
        ),

        limit_train_batches=(
            tc[
                "limit_train_batches"
            ]
        ),

        gradient_clip_val=1.0,

        default_root_dir=(
            output_dir
        ),

        logger=CSVLogger(
            save_dir=output_dir,
            name="",
            version="",
        ),

        callbacks=[
            ModelCheckpoint(
                dirpath=(
                    output_dir
                    / "checkpoints"
                ),
                save_top_k=0,
                save_last=True,
            )
        ],
    )

    trainer.fit(
        module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
