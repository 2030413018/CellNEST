# Running CellNEST on Single-Cell RNA-seq Data (Without Spatial Coordinates)

## Background

CellNEST was originally designed for spatial transcriptomics (ST) data, where each cell/spot has physical x-y coordinates used to define cell neighborhoods and weight cell-cell interactions by distance. However, the core idea — learning ligand-receptor communication patterns via a graph attention network — can also be applied to standard single-cell RNA-seq (scRNA-seq) data that lacks spatial information.

## How It Works

When using `--data_type=scrna`, CellNEST adapts its pipeline as follows:

| Aspect | Spatial Transcriptomics (original) | scRNA-seq (adapted) |
|--------|-----------------------------------|--------------------|
| **Coordinates** | Physical tissue positions | Derived from PCA + UMAP on expression |
| **Neighborhood** | Physical distance (fixed threshold or KNN in tissue space) | KNN in UMAP embedding space (forced) |
| **Edge weights** | Inverse physical distance | Inverse UMAP-space distance |
| **Juxtacrine (cell-contact) filtering** | Based on spot diameter | Disabled (no physical contact concept) |
| **All other steps** | Unchanged | Unchanged |

### Key Design Decisions

1. **UMAP-based neighborhoods**: Cells that are transcriptomically similar end up close in UMAP space and become neighbors. This is biologically reasonable since cells with similar expression profiles are more likely to share a microenvironment or lineage.

2. **KNN is forced**: The `fixed` distance mode relies on a physical distance threshold, which has no meaning in UMAP space. Therefore, `--distance_measure` is automatically set to `knn`.

3. **Juxtacrine interactions are disabled**: "Cell-Cell Contact" type interactions require physical proximity, which cannot be determined from expression alone. These are automatically excluded via `--block_juxtacrine=1`.

## Usage

### Step 1: Prepare your data

Your scRNA-seq data should be in `.h5ad` (AnnData) format with **raw counts** (not normalized or log-transformed). CellNEST will perform its own quantile normalization. Make sure to run QC and cell filtering beforehand.

### Step 2: Preprocess

Use `--data_type=scrna` with the standard `preprocess` command, or use the shorthand `preprocess_scrna`:

```bash
# Option A: explicit data_type flag
cellnest preprocess --data_name='my_scrna_dataset' \
    --data_from='path/to/my_data.h5ad' \
    --data_type='scrna'

# Option B: shorthand command (sets --data_type=scrna automatically)
cellnest preprocess_scrna --data_name='my_scrna_dataset' \
    --data_from='path/to/my_data.h5ad'
```

You can tune the number of neighbors with `--k` (default: 50). The value is automatically clamped to not exceed the number of cells minus 1.

### Step 3: Train, Postprocess, Visualize

These steps are identical to the standard spatial workflow:

```bash
# Train (repeat with different --run_id for ensemble)
cellnest run --data_name='my_scrna_dataset' \
    --num_epoch 80000 \
    --model_name='CellNEST_my_scrna' \
    --run_id=1

# Postprocess (ensemble multiple runs)
cellnest postprocess --data_name='my_scrna_dataset' \
    --model_name='CellNEST_my_scrna' \
    --total_runs=5

# Visualize
cellnest visualize --data_name='my_scrna_dataset' \
    --model_name='CellNEST_my_scrna'
```

## Caveats

- The 2D positions are derived from expression, not physical location. The resulting visualizations show cells in **expression space**, not tissue space.
- Communication predictions reflect transcriptomic similarity-based neighborhoods, not physical proximity. This is a fundamentally different biological assumption.
- For large datasets (>10,000 cells), consider using the [split graph option](split_graph_option.md) to manage memory.
- If you have both scRNA-seq and spatial data, consider using the [MERFISH integration workflow](integrate_scRNAseq_merfish.md) instead, which leverages spatial information from the imaging-based data.
