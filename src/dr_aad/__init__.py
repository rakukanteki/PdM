"""DR-AAD: a new implementation of the manuscript's method."""
import os

# Required by deterministic CUDA matrix multiplication; set before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
