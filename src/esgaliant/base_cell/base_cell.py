import cellxgene_census
import cellxgene_census.experimental.ml as census_ml
import tiledbsoma as soma
import torch
import zarr

# DEFINE UTIL FUNCIONS


# Make this function adaptable to other databases
def get_gene_set(
    atlas: str = "cellxgene",
    organism: str = "mus_musculus",
    max_size: None | int = None,
    version: str = "2025-01-30",
):
    if atlas == "cellxgene":
        with cellxgene_census.open_soma(census_version=version) as census:
            gene_metadata = cellxgene_census.get_var(
                census, organism, column_names=["feature_name", "feature_id"]
            )
            genes = gene_metadata["feature_name"].tolist()
        if max_size is None:
            return genes
        else:
            return genes[0:max_size]
    else:
        # Place holder for future data bases
        raise ValueError(f"Unknown atlas: {atlas}")


# Make this function adaptable to other databases
def get_cell_set(
    atlas: str = "cellxgene",
    organism: str = "mus_musculus",
    max_size: None | int = None,
    version: str = "2025-01-30",
):
    if atlas == "cellxgene":
        with cellxgene_census.open_soma(census_version=version) as census:
            cell_meta_data = cellxgene_census.get_obs(
                census, organism, column_names=["cell_type"]
            )
            cell_types = list(set(cell_meta_data["cell_type"]))
        if max_size is None:
            return cell_types
        else:
            return cell_types[0:max_size]
    else:
        raise ValueError(f"Unknown atlas: {atlas}")


# For clarity and ease this one will be data base specific
# we can just write a wrapper for these later if needed.
def base_cellxgene(
    gene_set,
    cell_set,
    organism: str = "mus_musculus",
    batch_size: int = 1000,
    version: str = "2025-01-30",
    device: str = "cpu",
):
    # Concat gene and cell set
    # always use primary data
    # User should always provide this data
    feature_set = "feature_name in ['" + "', '".join(gene_set) + "']"
    obs_set = (
        "is_primary_data == True and cell_type in ['"
        + "', '".join(cell_set)
        + "']"
    )

    # First we pull the data and use the pytorch loader.
    # So silly but at least we can GPU accelerate compute...
    with cellxgene_census.open_soma(census_version=version) as census:
        # get the minimal batch size
        query_size = cellxgene_census.get_obs(
            census, organism, value_filter=obs_set
        )
        effective_batch_size = min([batch_size, query_size.shape[0]])
        # get actual data to parse to experiment
        base_cell_data = census["census_data"][organism]
        # Check actual batch size - not the best but it will do for now
        obs_query = soma.AxisQuery(value_filter=obs_set)
        var_query = soma.AxisQuery(value_filter=feature_set)

        experiment_datapipe = census_ml.ExperimentDataPipe(
            base_cell_data,
            measurement_name="RNA",
            X_name="normalized",
            obs_query=obs_query,
            var_query=var_query,
            obs_column_names=["cell_type"],
            batch_size=effective_batch_size,
            shuffle=False,
        )

        # Compute mean using torch
        experiment_dataloaded = census_ml.experiment_dataloader(
            experiment_datapipe
        )
        # will need to make a check here to add gpu
        device = torch.device(device)
        mean = 0.0
        var = 0.0
        batch_count = 1
        print(f"Number of batches = {len(experiment_dataloaded)}")
        for gene_tensor, _ in experiment_dataloaded:
            print(f"Processing batch {batch_count}")
            local_gene_tensor = gene_tensor.to(device)
            local_mean = torch.mean(local_gene_tensor, axis=0)
            local_var = torch.var(local_gene_tensor, axis=0)
            mean += local_mean
            var += local_var
            total_mean = mean / batch_count
            total_mean = total_mean.detach().clone()
            total_var = var / batch_count
            total_var = total_var.detach().clone()
            batch_count += 1
    return total_mean.numpy(), total_var.numpy()


def get_base_cells(
    zarr_path,
    gene_set,
    cell_set,
    organism: str = "mus_musculus",
    batch_size: int = 1000,
    version: str = "2025-01-30",
    device: str = "cpu",
):
    # First we get THE base cell
    # Not going to
    base_mean, base_var = base_cellxgene(
        gene_set, cell_set, organism, batch_size, version, device
    )
    # write the results to zarr store
    zarr_store = zarr.open(store=zarr_path, mode="r+")
    zarr_store["base_cell_mean"][0, :] = base_mean
    zarr_store["base_cell_var"][0, :] = base_var
    # Next we run over each cell types to get the mean cell of each cell type
    for ct in range(1, len(cell_set) + 1):
        cell_type = cell_set[ct - 1]
        print(f"Current cell type = {cell_type}")
        local_mean, local_var = base_cellxgene(
            gene_set, [cell_type], organism, batch_size, version, device
        )
        zarr_store["base_cell_mean"][ct, :] = local_mean
        zarr_store["base_cell_var"][ct, :] = local_var
    return zarr_store


# # Doesn't currently work - get connection error and it is not clear what the "path to experiement should be."
# def base_cellxgene_newapi(
#     gene_set,
#     cell_set,
#     organism: str = "mus_musculus",
#     batch_size: int = 1000,
#     version: str = "2025-01-30",
#     device: str = "cpu",
# ):
#     # Build filter strings
#     gene_filter = "feature_name in ['" + "', '".join(gene_set) + "']"
#     cell_filter = (
#         "is_primary_data == True and cell_type in ['"
#         + "', '".join(cell_set)
#         + "']"
#     )

#     # Open CELLxGENE census and get the experiment path
#     with cellxgene_census.open_soma(census_version=version) as census:
#         exp = census["census_data"][organism]
#         experiment_uri = exp.uri

#     # Now open as TileDB-SOMA Experiment with the new API
#     with Experiment.open(experiment_uri) as exp:
#         # Use axis_query for filtering
#         with exp.axis_query(
#             measurement_name="RNA",
#             obs_query=AxisQuery(value_filter=cell_filter),
#             var_query=AxisQuery(value_filter=gene_filter),
#             obs_column_names=["cell_type"],
#         ) as query:
#             # Create dataset and dataloader
#             ds = ExperimentDataset(query, batch_size=batch_size)
#             experiment_dataloaded = experiment_dataloader(ds)

#             # Set up device and initialize accumulators
#             device_obj = torch.device(device)

#             total_mean = 0.0
#             total_var = 0.0
#             batch_count = 0

#             # Iterate through batches
#             for X_batch, obs_batch in experiment_dataloaded:
#                 # X_batch: expression matrix (n_obs, n_vars)
#                 # obs_batch: observation metadata
#                 X_batch = X_batch.to(device_obj)
#                 print(f"Batch shape: {X_batch.shape}")

#                 # Compute batch statistics
#                 batch_mean = X_batch.mean().item()
#                 batch_var = X_batch.var().item()

#                 # Accumulate
#                 total_mean += batch_mean
#                 total_var += batch_var
#                 batch_count += 1

#             # Average over batches
#             if batch_count > 0:
#                 mean = torch.tensor(total_mean / batch_count, device=device_obj)
#                 var = torch.tensor(total_var / batch_count, device=device_obj)
#             else:
#                 mean = torch.tensor(0.0, device=device_obj)
#                 var = torch.tensor(0.0, device=device_obj)

#     return mean, var
