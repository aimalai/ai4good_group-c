import argparse
import glob
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


class FineTuneModule(DualAuxDisturbanceModule):
    """Fresh low-LR optimizer; do NOT restore the old optimizer/scheduler."""

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.trainer.max_epochs)),
            eta_min=3e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def find_base_checkpoint():
    runs = sorted(
        glob.glob("runs/full_dual_ltae_s1s2_10ep_*")
    )

    if not runs:
        raise RuntimeError(
            "Could not find the completed Dual L-TAE 10-epoch run."
        )

    run = Path(runs[-1])
    ckpt = run / "checkpoints" / "last.ckpt"

    if not ckpt.exists():
        raise RuntimeError(f"Missing checkpoint: {ckpt}")

    return run, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(
        "configs/statistical_mlp.yaml",
        args.overrides,
    )

    config["model"] = {
        "name": "dual_ltae_s1s2_finetune",
        "base_run": "auto-detected",
        "s2_dim": 128,
        "s1_dim": 96,
        "s2_heads": 8,
        "s1_heads": 4,
        "fusion_dim": 512,
        "dropout": 0.10,
        "s2_acquisition_dropout": 0.05,
        "s1_acquisition_dropout": 0.05,
        "aux_weight_each": 0.10,
        "finetune_lr": 3e-5,
    }

    base_run, ckpt_path = find_base_checkpoint()

    print("Base run:", base_run)
    print("Loading weights:", ckpt_path)

    output_dir = (
        Path(config["output_root"])
        / f"{config['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config["model"]["base_run"] = str(base_run)
    config["model"]["base_checkpoint"] = str(ckpt_path)

    save_config(config, output_dir / "config.yaml")

    print("Fine-tune run folder:", output_dir)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(config["seed"], workers=True)

    mapping = build_mapping(config)
    datamodule = build_datamodule(config, mapping)

    model = DualLTAEFusion(
        num_classes=mapping.num_classes,
        s2_dim=128,
        s1_dim=96,
        s2_heads=8,
        s1_heads=4,
        fusion_dim=512,
        dropout=0.10,
        s2_acquisition_dropout=0.05,
        s1_acquisition_dropout=0.05,
    )

    loss = RecentDisturbanceLoss(
        no_disturbance_id=mapping.no_disturbance_id,
        **config["loss"],
    )

    module = FineTuneModule(
        model,
        loss,
        mapping,
        lr=3e-5,
        weight_decay=config["optimizer"]["weight_decay"],
        output_dir=output_dir,
        evaluation=config["evaluation"],
        aux_weight=0.10,
        eta_min=3e-6,
    )

    # Load MODEL/MODULE WEIGHTS ONLY.
    # We intentionally do not pass ckpt_path to trainer.fit(),
    # because that would restore the old optimizer and LR schedule.
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    module.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    print("Checkpoint weights loaded successfully.")
    print("Fine-tune LR: 3e-5")
    print("Auxiliary loss weight per branch: 0.10")
    print("S2 acquisition dropout: 0.05")
    print("S1 acquisition dropout: 0.05")

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
                filename="finetune-{epoch:02d}",
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
