# -----------------------------------------------------------------------------#
# Cell population
# -----------------------------------------------------------------------------#


# @dataclass(slots=True)
# class CellState:
#     cycle_position: np.int32
#     coordinates: np.ndarray  # float32, shape (3,)
#     cell_info: dict | None = None
#     ecosystem: np.ndarray | None = None  # float32
#     chromatin_state: csr_matrix | None = None
#     binding_state: csr_matrix | None = None
#     rna_state: np.ndarray | None = None  # int32
#     protein_state: np.ndarray | None = None  # int32
#     metabolome_state: csr_matrix | None = None
#     messaging_state: csr_matrix | None = None

#     def __post_init__(self):
#         self.cycle_position = np.int32(self.cycle_position)
#         self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
#         if self.ecosystem is not None:
#             self.ecosystem = np.asarray(self.ecosystem, dtype=np.float32)
#         if self.rna_state is not None:
#             self.rna_state = np.asarray(self.rna_state, dtype=np.int32)
#         if self.protein_state is not None:
#             self.protein_state = np.asarray(self.protein_state, dtype=np.int32)


def initialize_cell_population():
    return 0
