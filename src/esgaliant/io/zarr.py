import numpy as np
import zarr


def initialize_store(
    path,
    n_samples: int = 1,
    n_cells: int = 5000,
    n_genes: int = 2000,
    modalities: list = ["rna"],
    add_spatial: bool = True,
    add_labels: bool = True,
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
    # Create the root zarr store
    root = zarr.open_group(path, mode="w")

    # Create hierarchy: root/sample_X/modality/data
    for sample_idx in range(n_samples):
        sample_name = f"sample_{sample_idx}"
        sample_group = root.create_group(sample_name)

        for modality in modalities:
            modality_group = sample_group.create_group(modality)

            # Create the main data array: cells x genes, filled with zeros
            modality_group.create_array(
                "data",
                shape=(n_cells, n_genes),
                dtype=np.float32,
                chunks=(min(1000, n_cells), min(1000, n_genes)),
                fill_value=0.0,
            )

            # Create an 'obs' group for cell-level metadata (observations)
            obs_group = modality_group.create_group("obs")

            # Add spatial coordinates if requested
            if add_spatial:
                obs_group.create_array(
                    "spatial_coords",
                    shape=(n_cells, 2),
                    dtype=np.float32,
                    chunks=(min(1000, n_cells), 2),
                    fill_value=0.0,
                )

            # Add cell type labels if requested
            if add_labels:
                # Initialize with empty strings
                labels = np.array([""] * n_cells, dtype=object)
                obs_group.create_array(
                    "cell_types",
                    data=labels,
                    chunks=(min(1000, n_cells),),
                    object_codec=zarr.codecs.JSON(),
                )

            # Create a 'var' group for gene-level metadata (variables)
            var_group = modality_group.create_group("var")

            # Initialize gene names
            gene_names = np.array(
                [f"gene_{i}" for i in range(n_genes)], dtype=object
            )
            var_group.create_array(
                "gene_names",
                data=gene_names,
                chunks=(min(1000, n_genes),),
                object_codec=zarr.codecs.JSON(),
            )

            # Add scalar metadata as attributes
            modality_group.attrs["n_cells"] = n_cells
            modality_group.attrs["n_genes"] = n_genes

    # Add root-level metadata
    root.attrs["n_samples"] = n_samples
    root.attrs["n_cells"] = n_cells
    root.attrs["n_genes"] = n_genes
    root.attrs["modalities"] = modalities

    return root


# Example usage
if __name__ == "__main__":
    # Create a zarr store with spatial coordinates and cell type labels
    store = initialize_store(
        "my_data.zarr",
        n_samples=3,
        n_cells=5000,
        n_genes=2000,
        modalities=["rna", "atac"],
        add_spatial=True,
        add_labels=True,
    )

    # Access data
    print(f"Store structure: {list(store.group_keys())}")
    print(f"Sample 0 RNA data shape: {store['sample_0/rna/data'].shape}")
    print(
        f"Spatial coords shape: {store['sample_0/rna/obs/spatial_coords'].shape}"
    )
    print(f"Cell types shape: {store['sample_0/rna/obs/cell_types'].shape}")
    print(f"Gene names shape: {store['sample_0/rna/var/gene_names'].shape}")

    # Example: Update some values
    # Add actual spatial coordinates for first 10 cells
    store["sample_0/rna/obs/spatial_coords"][:10, :] = (
        np.random.Generator(10, 2) * 100
    )

    # Add cell type labels for first 10 cells
    store["sample_0/rna/obs/cell_types"][:10] = [
        "T-cell",
        "B-cell",
        "T-cell",
        "Macrophage",
        "T-cell",
        "B-cell",
        "Neuron",
        "T-cell",
        "Macrophage",
        "B-cell",
    ]

    print(
        f"\nFirst 5 spatial coords:\n{store['sample_0/rna/obs/spatial_coords'][:5]}"
    )
    print(f"First 10 cell types: {store['sample_0/rna/obs/cell_types'][:10]}")
