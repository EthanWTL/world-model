# Locations: only from the environment (run.sh sets them from config.env; see INSTRUCTIONS 2.3)
#   VM_DATA_DIR: the folder that holds RubikMemory/   VM_OUT_DIR: every run's outputs   VM_VAE: the Wan2.1 VAE folder
import os

DATA_DIR = os.environ["VM_DATA_DIR"]
OUT_DIR = os.environ["VM_OUT_DIR"]
VAE = os.environ["VM_VAE"]
