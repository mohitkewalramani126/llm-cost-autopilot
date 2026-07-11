from pathlib import Path
import yaml

from app.models import ModelConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


def load_registry(path: Path = CONFIG_PATH) -> list[ModelConfig]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return [ModelConfig(**entry) for entry in data["models"]]


MODEL_REGISTRY = load_registry()