| Model | Test R2 | Test RMSE | Test MAE | Akay Spearman log10 | Akay RMSE log10 | Akay MAPE % | Akay within 2x | Top-100 overlap | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Frozen embeddings+NN+availability tuned | 0.702 | 567590.625 | 369376.906 | 0.745 | 0.331 | 64.921 | 0.645 | 0.530 | selected final model |
| Frozen embeddings+NN+availability domain-weighted | 0.693 | 576602.812 | 369912.531 | 0.753 | 0.333 | 59.282 | 0.641 | 0.520 | soft target-domain weighting toward table_3 |
| Frozen embeddings+NN+availability | 0.626 | 635703.000 | 406633.500 | 0.737 | 0.369 | 64.739 | 0.597 | 0.480 | availability added |
| Frozen embeddings+NN | 0.335 | 848327.750 | 588749.062 | 0.487 | 0.464 | 126.314 | 0.543 | 0.440 | best pre-availability frozen baseline |
| Fine-tuned bert+NN+availability retry | 0.495 | 738817.000 | 509965.656 | 0.631 | 0.416 | 95.172 | 0.567 | 0.450 | clean retry with gentler 1-layer fine-tune |
| Fine-tuned bert+NN+availability | 0.498 | 737127.875 | 504695.625 | 0.617 | 0.457 | 83.097 | 0.537 | 0.440 | first availability-aware fine-tune |
| Fine-tuned bert+NN+availability staged | 0.467 | 759582.875 | 521637.906 | 0.639 | 0.426 | 86.107 | 0.556 | 0.450 | 2-epoch head warmup then 1-layer unfreeze |
| Fine-tuned bert+NN best | 0.275 | 885519.750 | 637536.875 | 0.474 | 0.484 | 130.385 | 0.496 | 0.430 | best fine-tune before availability |
