import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

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

from experiments.tristan_s1s2_fusion.model import (
    TesseraS1S2Fusion,
)


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
        "name": "tessera_s1s2_late_fusion",
        "s1_frame_dim": 64,
        "s1_hidden_dim": 128,
        "head_dim": 512,
        "dropout": 0.10,
    }

    output_dir = (
        Path(config["output_root"])
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
        output_dir / "config.yaml",
    )

    print("Run folder:", output_dir)

    torch.set_float32_matmul_precision(
        "high"
    )

    L.seed_everything(
        config["seed"],
        workers=True,
    )

    mapping = build_mapping(config)

    datamodule = build_datamodule(
        config,
        mapping,
    )

    model = TesseraS1S2Fusion(
        num_classes=mapping.num_classes,
        s1_frame_dim=64,
        s1_hidden_dim=128,
        head_dim=512,
        dropout=0.10,
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )

    print(
        f"Frozen TESSERA parameters: {frozen:,}"
    )

    loss = RecentDisturbanceLoss(
        no_disturbance_id=mapping.no_disturbance_id,
        **config["loss"],
    )

    module = DisturbanceModule(
        model,
        loss,
        mapping,
        **config["optimizer"],
        output_dir=output_dir,
        evaluation=config["evaluation"],
    )

    tc = config["trainer"]

    trainer = L.Trainer(
        accelerator=tc["accelerator"],
        devices=tc["devices"],
        precision=tc["precision"],
        max_epochs=tc["max_epochs"],
        limit_train_batches=tc[
            "limit_train_batches"
        ],
        default_root_dir=output_dir,
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
