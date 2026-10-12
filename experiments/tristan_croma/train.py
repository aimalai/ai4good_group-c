import argparse
from datetime import datetime
from pathlib import Path

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

from experiments.tristan_croma.model import (
    FrozenCROMAClassifier,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "overrides",
        nargs="*",
    )

    args = parser.parse_args()

    # Start directly from professor baseline config so
    # loss/evaluation settings remain identical.
    config = load_config(
        "configs/statistical_mlp.yaml",
        args.overrides,
    )

    config["model"] = {
        "name": "frozen_croma",
        "projection_dim": 128,
        "trunk_dim": 256,
        "encoder_batch_size": 16,
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

    L.seed_everything(
        config["seed"],
        workers=True,
    )

    mapping = build_mapping(config)

    datamodule = build_datamodule(
        config,
        mapping,
    )

    model = FrozenCROMAClassifier(
        num_classes=mapping.num_classes,
        projection_dim=128,
        trunk_dim=256,
        encoder_batch_size=16,
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
        f"Frozen CROMA parameters: {frozen:,}"
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
        default_root_dir=output_dir,
    )

    trainer.fit(
        module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
