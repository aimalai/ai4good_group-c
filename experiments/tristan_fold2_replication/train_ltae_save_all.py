import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from forest_disturbance.build import build_datamodule, build_mapping
from forest_disturbance.config import load_config, save_config
from forest_disturbance.models.loss import RecentDisturbanceLoss

from experiments.tristan_ltae_late_fusion.model import DualLTAEFusion
from experiments.tristan_ltae_late_fusion.train import DualAuxDisturbanceModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
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
        "checkpoint_policy": "save every epoch; select by validation Macro F1 @ 60d",
    }

    output_dir = (
        Path(config["output_root"])
        / f"{config['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config(
        config,
        output_dir / "config.yaml",
    )

    print("Run folder:", output_dir)

    torch.set_float32_matmul_precision("high")

    L.seed_everything(
        config["seed"],
        workers=True,
    )

    mapping = build_mapping(config)
    datamodule = build_datamodule(
        config,
        mapping,
    )

    model = DualLTAEFusion(
        num_classes=mapping.num_classes,
        s2_dim=128,
        s1_dim=96,
        s2_heads=8,
        s1_heads=4,
        fusion_dim=512,
        dropout=0.10,
        s2_acquisition_dropout=0.20,
        s1_acquisition_dropout=0.10,
    )

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    loss = RecentDisturbanceLoss(
        no_disturbance_id=mapping.no_disturbance_id,
        **config["loss"],
    )

    module = DualAuxDisturbanceModule(
        model,
        loss,
        mapping,
        **config["optimizer"],
        output_dir=output_dir,
        evaluation=config["evaluation"],
        aux_weight=0.25,
        eta_min=3e-6,
    )

    tc = config["trainer"]

    trainer = L.Trainer(
        accelerator=tc["accelerator"],
        devices=tc["devices"],
        precision=tc["precision"],
        max_epochs=tc["max_epochs"],
        limit_train_batches=tc["limit_train_batches"],
        gradient_clip_val=1.0,
        default_root_dir=output_dir,

        logger=CSVLogger(
            save_dir=output_dir,
            name="",
            version="",
        ),

        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename="epoch-{epoch:02d}",
                auto_insert_metric_name=False,
                save_top_k=-1,
                every_n_epochs=1,
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
