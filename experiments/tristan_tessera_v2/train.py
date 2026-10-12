import argparse
from datetime import datetime
from pathlib import Path

import torch
import lightning as L

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

from experiments.tristan_tessera_v2.model import (
    FrozenTesseraV2,
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
        "name": "frozen_tessera_v2_medium_s2",
        "hidden_dim": 256,
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

    model = FrozenTesseraV2(
        num_classes=mapping.num_classes,
        hidden_dim=256,
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

    trainer_config = config["trainer"]

    trainer = L.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        precision=trainer_config["precision"],
        max_epochs=trainer_config["max_epochs"],
        limit_train_batches=trainer_config[
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
                dirpath=output_dir / "checkpoints",
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
