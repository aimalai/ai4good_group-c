"""Train the isolated TemporalGRU experiment without modifying course baseline files.

Run from the repository root, for example:

    python experiments/tristan_temporal_gru/train.py \
        trainer.max_epochs=1 trainer.limit_train_batches=200 logger.kind=csv

The script deliberately reuses the course data module, target construction, loss,
Lightning wrapper, prediction writer and evaluation metrics. Only the model is new.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from forest_disturbance.build import REPO_ROOT, build_datamodule, build_mapping
from forest_disturbance.config import load_config, save_config
from forest_disturbance.models.lit_module import DisturbanceModule
from forest_disturbance.models.loss import RecentDisturbanceLoss

from model import TemporalGRU


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("overrides", nargs="*", help="key.subkey=value")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    output_dir = (
        Path(config["output_root"]) / f"{config['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True)
    save_config(config, output_dir / "config.yaml")
    print(f"Run folder: {output_dir}")

    L.seed_everything(config["seed"], workers=True)
    mapping = build_mapping(config)
    datamodule = build_datamodule(config, mapping)

    model_args = {key: value for key, value in config["model"].items() if key != "name"}
    model = TemporalGRU(num_classes=mapping.num_classes, **model_args)
    loss = RecentDisturbanceLoss(
        no_disturbance_id=mapping.no_disturbance_id,
        **config["loss"],
    )
    module = DisturbanceModule(
        model=model,
        loss=loss,
        mapping=mapping,
        **config["optimizer"],
        output_dir=output_dir,
        evaluation=config["evaluation"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TemporalGRU trainable parameters: {trainable:,}")

    trainer_config = config["trainer"]
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            save_top_k=0,
            save_last=True,
        )
    ]
    if trainer_config["early_stopping_patience"] is not None:
        callbacks.append(
            EarlyStopping(
                monitor=trainer_config["early_stopping_monitor"],
                mode="max",
                patience=trainer_config["early_stopping_patience"],
            )
        )

    trainer = L.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        precision=trainer_config["precision"],
        max_epochs=trainer_config["max_epochs"],
        limit_train_batches=trainer_config["limit_train_batches"],
        default_root_dir=output_dir,
        logger=make_logger(config, output_dir),
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule)


def make_logger(config: dict, output_dir: Path):
    """Mirror scripts/train.py so experiment output stays directly comparable."""
    logger = config["logger"]
    if logger["kind"] == "wandb":
        return WandbLogger(
            name=output_dir.name,
            project=logger["project"],
            entity=logger["entity"],
            offline=logger["offline"],
            save_dir=output_dir,
            config=config,
        )
    if logger["kind"] == "csv":
        return CSVLogger(save_dir=output_dir, name="", version="")
    if logger["kind"] == "none":
        return False
    raise ValueError(f"Unknown logger.kind {logger['kind']!r}: use wandb, csv or none.")


if __name__ == "__main__":
    main()
