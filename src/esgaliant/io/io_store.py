import numpy as np
import zarr


def compute_chunk_size(
    n_cells: int = 10000,
    n_genes: int = 2000,
    max_chunk_size: int = 10000,
    min_chunk_size: int = 1000,
):
    print(n_genes)
    print(min_chunk_size)
    if n_genes < min_chunk_size:
        raise ValueError(
            "Number of genes should not be smaller than min chunk size."
        )
    if n_cells < min_chunk_size:
        raise ValueError(
            "Number of cells should not be smaller than min chunk size."
        )
    gene_chunk_size = min(max_chunk_size, n_genes)
    cell_chunk_size = min(max_chunk_size, n_cells)
    return gene_chunk_size, cell_chunk_size


def initialize_store(
    path,
    n_samples: int = 1,
    n_cells: int = 10000,
    n_genes: int = 2000,
    max_chunk_size: int = 10000,
    min_chunk_size: int = 1000,
    modalities: list = ["rna"],
    add_spatial: bool = True,
):
    """
    Initialize a hierarchical zarr store with samples containing cell x gene matrices.

    Parameters
    ----------
    path : str
        Path to the zarr store
    n_samples : int
        Number of samples to create
    n_cells : int
        Number of cells (rows) per sample
    n_genes : int
        Number of genes (columns) per sample
    max_chunk_size : int
        Max size for a zarr chunk - if the number of cells or genes are high
    modalities : list
        List of modality names (e.g., ["rna", "atac"])
    add_spatial : bool
        If True, initialize spatial coordinates (filled with zeros)
    add_labels : bool
        If True, initialize cell type labels (filled with empty strings)

    Returns
    -------
    zarr.hierarchy.Group
        The root zarr group
    """
    # compute chunk size bounds
    gene_chunk_size, cell_chunk_size = compute_chunk_size(
        n_cells, n_genes, max_chunk_size, min_chunk_size
    )
    # Create the root zarr store
    store = zarr.storage.LocalStore(path, read_only=False)
    root = zarr.create_group(store)

    # Create hierarchy: root/sample_X/modality/data
    for sample_idx in range(n_samples):
        sample_name = f"sample_{sample_idx}"
        sample_group = root.create_group(sample_name)

        for modality in modalities:
            # Which modality are we looking at
            modality_group = sample_group.create_group(modality)

            # Create empty array for one modality.
            modality_group.create_array(
                "data",
                shape=(n_cells, n_genes),
                dtype=np.float32,
                chunks=(cell_chunk_size, gene_chunk_size),
                fill_value=0.0,
            )

            # Create an 'obs' group for cell-level metadata (observations)
            # Auto filling with cell ids
            obs_group = modality_group.create_group("obs")
            cell_ids = np.array([f"cell_{i}" for i in range(n_cells)])
            obs_group.create_array(
                "cell_ids",
                data=cell_ids,
                chunks=(cell_chunk_size,),
            )
            # Add spatial coordinates if requested - can only be single cell data
            # but this will be very rarely the case since we will include spatial effects
            # even for single cell data.
            if add_spatial:
                obs_group.create_array(
                    "spatial_coords",
                    shape=(n_cells, 3),
                    dtype=np.float32,
                    chunks=(cell_chunk_size, 3),
                    fill_value=0.0,
                )

            # Create a 'var' group for gene-level metadata (variables)
            var_group = modality_group.create_group("var")

            # Initialize gene names
            gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
            var_group.create_array(
                "gene_names",
                data=gene_names,
                chunks=(gene_chunk_size,),
            )

            # Add scalar metadata as attributes
            modality_group.attrs["n_cells"] = n_cells
            modality_group.attrs["n_genes"] = n_genes
            modality_group.attrs["spatial_coord"] = add_spatial

    # Add root-level metadata
    root.attrs["n_samples"] = n_samples
    root.attrs["n_cells"] = n_cells
    root.attrs["n_genes"] = n_genes
    root.attrs["modalities"] = modalities
    root.attrs["spatial_coord"] = add_spatial
    return root


# Example usage
if __name__ == "__main__":
    # Create a zarr store with spatial coordinates and cell type labels
    store = initialize_store(
        "/Users/martinp4/Documents/Cedars/esgaliant/esgaliant/data/test.zarr",
        n_samples=3,
        n_cells=5000,
        n_genes=2000,
        modalities=["rna", "atac"],
        add_spatial=True,
    )

    # Access data
    print(f"Store structure: {list(store.group_keys())}")
    print(f"Sample 0 RNA data shape: {store['sample_0/rna/data'].shape}")
    print(
        f"Spatial coords shape: {store['sample_0/rna/obs/spatial_coords'].shape}"
    )
    print(f"Cell types shape: {store['sample_0/rna/obs/cell_ids'].shape}")
    print(f"Gene names shape: {store['sample_0/rna/var/gene_names'].shape}")
