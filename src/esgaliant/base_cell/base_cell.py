import cellxgene_census
import numpy as np
import numpy.typing as npt
import tiledbsoma as soma
import zarr

from esgaliant.io.io_store import compute_chunk_size


def base_cell(
    atlas: list = ["cellxgene"],
    organism: str = "mus_musculus",
    cell_types: None | list[str] = None,
    zarr_path: str = ".",  # Not sure if I will use zarr path here or not
    chunk_size: int = 1000,
):
    gene_set = get_common_genes(atlas, organism)
    base_cells = []
    for at in atlas:
        if at == "cellxgene":
            base_cells.append(base_cellxgene(gene_set, organism, cell_types))
        elif at == "other":
            print("none at the moment")
        else:
            raise ValueError(f"Unknown atlas: {at}")
    # Leaving space to using co-embedding space to run this on all atlases
    return base_cells


def get_common_genes(
    atlas: list = ["cellxgene"],
    organism: str = "mus_musculus",
):
    gene_sets = []

    for at in atlas:
        if at == "cellxgene":
            with cellxgene_census.open_soma() as census:
                genes = set(census["census_data"][organism].ms["RNA"].var.index)
                gene_sets.append(genes)
        elif at == "other_atlas":
            # Place holder
            genes = set(...)
            gene_sets.append(genes)
        else:
            raise ValueError(f"Unknown atlas: {at}")

    if not gene_sets:
        raise ValueError("No valid atlases provided")

    if len(gene_sets) == 1:
        return gene_sets[0]
    else:
        return set.intersection(*gene_sets)


def base_cellxgene(
    gene_set: list[str],
    organism: str = "mus_musculus",
    cell_types: list[str] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Compute mean expression for a gene set, optionally filtered by cell types.

    Parameters:
    -----------
    gene_set : list[str]
        Compulsory. List of gene names to compute mean expression for.
    organism : str
        Organism name (default: "mus_musculus")
    cell_types : list[str] | None
        Optional. List of cell type values to filter on. If None, uses all cell types.

    Returns:
    --------
    npt.NDArray[np.float64]
        Mean expression values for the gene set across selected cells.
    """
    # connect to cellxgene server
    with cellxgene_census.open_soma() as census:
        org = census["census_data"][organism]

        # Build obs_query with primary data filter
        obs_filter = "is_primary_data==True"

        # Add cell type filter if provided
        if cell_types is not None:
            cell_type_filter = " && ".join(
                [f'cell_type=="{ct}"' for ct in cell_types]
            )
            obs_filter = f"({obs_filter}) && ({cell_type_filter})"

        # Always filter primary tissue to remove duplicate cells
        with org.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_filter),
            var_query=soma.AxisQuery(
                value_filter=" || ".join(
                    [f'feature_name=="{gene}"' for gene in gene_set]
                )
            ),
        ) as query:
            # Get variable information
            var_df = query.var().concat().to_pandas()
            n_vars = len(var_df)
            n_obs = query.n_obs

            print(f"Filtering for {n_obs} cells and {n_vars} genes")

            # Initialize accumulators for per-gene means
            gene_sum = np.zeros((n_vars,), dtype=np.float64)
            gene_count = np.zeros((n_vars,), dtype=np.int64)

            # Get indexer to map soma_joinid to positional indices
            indexer = query.indexer

            # Stream through X data in batches
            for chunk_idx, arrow_tbl in enumerate(query.X("raw").tables()):
                print(f"Processing chunk {chunk_idx + 1}...")

                # Get positional indices for genes (var dimension)
                var_pos = indexer.by_var(arrow_tbl["soma_dim_1"])
                # Get the data values
                data = arrow_tbl["soma_data"].to_numpy()

                # Accumulate sums and counts per gene
                np.add.at(gene_sum, var_pos, data)
                np.add.at(gene_count, var_pos, 1)

                print(
                    f"  Chunk {chunk_idx + 1} complete: {len(data)} values processed"
                )

            # Compute final means
            gene_mean = np.divide(
                gene_sum,
                n_obs,
                where=(gene_count > 0),
                out=np.zeros_like(gene_sum),
            )
    return gene_mean


def cell_diff(
    gene_mean: npt.NDArray,
    cell_types: None | list[str],
    zarr_path: str = ".",
    chunk_size: int = 1000,
    keep_all: bool = False,
    atlas: str | list = "cellxgene",
):
    if "cellxgene" in atlas:
        with cellxgene_census.open_soma() as census:
            cell_meta_data = cellxgene_census.get_obs(
                census, "mus_musculus", column_names=["cell_type"]
            )
            if keep_all:
                cell_meta_data = list(cell_meta_data)
            else:
                cell_meta_data = set(cell_meta_data["cell_type"])

    return 0


def initialize_base_cell_store(
    gene_set_size,
    cell_set_size,
    zarr_path: str = ".",
    max_chunk_size: int = 5000,
    min_chunk_size: int = 1000,
    atlas: str = "cellxgene",
):
    store = zarr.storage.LocalStore(zarr_path, read_only=False)
    root = zarr.create_group(store)
    for at in atlas:
        cell_chunk_size, gene_chunk_size = compute_chunk_size(
            cell_set_size, gene_set_size, max_chunk_size, min_chunk_size
        )
        atlas_group = root.create_group(f"atlast_{at}")
        atlas_group.create_array(
            "base_cell",
            shape=(cell_set_size, gene_set_size),
            dtype=np.float32,
            chunks=(cell_chunk_size, gene_chunk_size),
            fill_value=0.0,
        )
        cell_id = atlas_group.create_group("obs")
        cell_id.create_array(
            "cell_types",
            shape=(cell_set_size,),
            dtype=str,
            chunks=(cell_chunk_size,),
        )
        gene_id = atlas_group.create_group("var")
        gene_id.create_array(
            "gene_names",
            shape=(gene_set_size,),
            dtype=str,
            chunks=(gene_chunk_size,),
        )
        atlas_group.attrs["n_cells"] = cell_set_size
        atlas_group.attrs["n_genes"] = gene_set_size

    return root
