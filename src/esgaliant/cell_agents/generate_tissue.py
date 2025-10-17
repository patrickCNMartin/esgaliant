from dataclasses import dataclass, field

import numpy as np
import zarr

from esgaliant.io import io_store


@dataclass(slots=True)
class TissueState:
    zarr_path: str = "esgaliant_atlas.zarr"
    n_samples: np.int32 = 1
    n_cells: np.int32 = 5000
    n_genes: np.int32 = 2000
    coordinate_range: np.ndarray = field(default_factory=[0, 1])
    modalities: list[str] = field(default_factory=lambda: ["rna"])
    time_step: np.int32 = 0
    max_chunk_size: int = 10000
    min_chunk_size: int = 1000
    store: zarr.Group | None = field(default=None, init=False)

    def __post_init__(self):
        self.store = io_store.initialize_store(
            path=self.zarr_path,
            n_samples=self.n_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            max_chunk_size=self.max_chunk_size,
            min_chunk_size=self.min_chunk_size,
            modalities=self.modalities,
        )
