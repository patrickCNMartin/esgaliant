import matplotlib.pyplot as plt
import seaborn as sns
import zarr

MAX_GENES = 100


def view_base_cells(zarr_path, modality: str = "base_cell_mean"):
    zarr_store = zarr.open(zarr_path, mode="r")

    mean_cells = zarr_store[modality][:]
    cell_types = zarr_store["cell_id"][:]
    features = zarr_store["features"][:]

    plt.figure(figsize=(10, 8))

    # Determine yticklabels based on the number of features
    if len(features) > MAX_GENES:
        ytick_labels = False
    else:
        ytick_labels = features

    ax = sns.heatmap(
        mean_cells,
        cmap="viridis",
        xticklabels=cell_types,
        yticklabels=ytick_labels,  # Use the determined value here
        annot=True,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Set yticklabels rotation only if labels are present
    if ytick_labels is not False:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title("Heatmap of Zarr Array Data")
    plt.xlabel("Cell Types")
    plt.ylabel("Features")

    plt.show()
