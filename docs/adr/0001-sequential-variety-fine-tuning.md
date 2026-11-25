# Add new Varieties by sequential fine-tuning, without replay

One model serves every Variety, each marked by its own Variety token. To add a Variety, we continue fine-tuning the latest published multi-Variety model on that Variety's Corpus alone, without mixing in earlier Corpora. The new Variety token is seeded from an existing Variety token (Santiago's `kea_st_XX` from São Vicente's `kea_sv_XX`). The Variety tokens are what separate the languages inside the model, and this reuses earlier fine-tuning without needing multi-Corpus training code.

## Considered Options

- **Replay**: continue from the published model, mixing the new Corpus with earlier ones. This protects earlier Varieties but needs multi-Corpus, multi-token training.
- **Joint from base**: train every Variety together starting from base NLLB. This throws away earlier fine-tuning.

## Consequences

The encoder and decoder weights are shared, so earlier Varieties can degrade (catastrophic forgetting). Every new model must therefore be evaluated on each earlier Variety's test set and compared with its parent before it is published. If it regresses, the first remedy is a lower learning rate, and the next is to revisit replay.
