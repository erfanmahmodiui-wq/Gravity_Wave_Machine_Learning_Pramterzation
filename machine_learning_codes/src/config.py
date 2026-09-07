import yaml
from pathlib import Path


def load_paths(config_path=None):
    """
    Load the directory/path config used throughout src/.

    By default looks for configs/paths.yaml relative to the repo root
    (i.e. one level up from this file's parent, src/).
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "paths.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
