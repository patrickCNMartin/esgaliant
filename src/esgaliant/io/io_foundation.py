from pathlib import Path

import numpy as np
import zarr


# Initialize store based on gene_set and cell set
def initialize_store(
    gene_set: int,
    cell_set: int,
    zarr_path: str | None = None,
    max_chunk_size: int = 10000,
):
    # define chunk size
    length_cell_set = len(cell_set)
    length_gene_set = len(gene_set)
    cell_chunk_size = min([max_chunk_size, length_cell_set])
    gene_chunk_size = min([max_chunk_size, length_gene_set])
    # Create the root zarr store
    if zarr_path is None:
        store = zarr.storage.MemoryStore()
    else:
        if Path(zarr_path).exists():
            print(f"{zarr_path} already exists - Skipping Store build")
            return zarr_path
        store = zarr.storage.LocalStore(zarr_path, read_only=False)
    root = zarr.create_group(store)

    # create base array for storing cell distances
    # Add a plus one to the cell set since we will store
    # the global cell average and each cell type average
    root.create_array(
        "base_cell_mean",
        shape=(length_cell_set + 1, length_gene_set),
        dtype=np.float64,
        chunks=(cell_chunk_size, gene_chunk_size),
        fill_value=0.0,
    )
    root.create_array(
        "base_cell_var",
        shape=(length_cell_set + 1, length_gene_set),
        dtype=np.float64,
        chunks=(cell_chunk_size, gene_chunk_size),
        fill_value=0.0,
    )
    cell_set = ["base_cell", *cell_set]
    root.create_array(name="cell_id", data=np.array(cell_set, dtype=str))
    root.create_array(name="features", data=np.array(gene_set, dtype=str))
    root.attrs["n_cells"] = length_cell_set
    root.attrs["n_features"] = length_gene_set

    return root
