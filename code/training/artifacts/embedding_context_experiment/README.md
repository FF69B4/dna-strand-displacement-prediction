Scratch token-level DNA-BERT context experiment.

This folder is intentionally separate from the main paper graph/data artifacts.
It reads existing table 3 sequences and a locally cached DNA-BERT model, then writes experiment-only outputs here.

- Target 6-mer: `GGGAGG`
- Token occurrences plotted: `100`
- PCA variance in first three components: `0.5437`
- `same_6mer_token_context_pca_3d.png`: 3D PCA plot of token hidden states grouped by immediate left/right sequence context.
- `same_6mer_token_context_vectors.csv`: extracted token metadata and PCA coordinates.
