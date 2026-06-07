from pathlib import Path
from utils import get_supported_embedders, DEFAULT_PROVIDER

PATH = Path(__file__).parents[0]
EMBEDDERS = get_supported_embedders(DEFAULT_PROVIDER)
DEFAULT_WALLTIME_LIMIT = 60 * 10 # 600 seconds
MIN_CHUNK_TOKEN_SIZE = 100
MAX_CHUNK_TOKEN_SIZE = 500
MIN_OVERLAP_PERCENTAGE = 0.1
MAX_OVERLAP_PERCENTAGE = 0.5
