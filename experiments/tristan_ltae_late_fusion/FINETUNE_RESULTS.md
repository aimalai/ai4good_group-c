# Dual L-TAE Fine-Tuning

- Original run: `runs/full_dual_ltae_s1s2_10ep_20261021-212830`
- Fine-tune run: `runs/finetune_dual_ltae_s1s2_5ep_20261022-095110`

Fine-tuning starts from the completed 10-epoch Dual L-TAE weights.
The old optimizer/scheduler is NOT restored.

- Additional epochs: 5
- Learning rate: 3e-5 with cosine decay
- Auxiliary branch loss weight: 0.10 each
- S2 acquisition dropout: 0.05
- S1 acquisition dropout: 0.05

## Best Macro F1 @ 60d

- Original: 0.473
- Fine-tuned: 0.459
- Change: -0.014

The fine-tuned checkpoint is selected only for comparison using Macro F1 @ 60d; all operational metrics should still be reported.
