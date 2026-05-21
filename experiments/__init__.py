from pathlib import Path
from utils import get_supported_embedders, DEFAULT_PROVIDER

PATH = Path(__file__).parents[0]
EMBEDDERS = get_supported_embedders(DEFAULT_PROVIDER)
