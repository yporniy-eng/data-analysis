"""Configuration loader"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Find config.yaml
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    config["database"]["clickhouse"]["host"] = os.getenv(
        "CLICKHOUSE_HOST", config["database"]["clickhouse"]["host"]
    )
    config["database"]["clickhouse"]["port"] = int(
        os.getenv("CLICKHOUSE_PORT", config["database"]["clickhouse"]["port"])
    )
    config["database"]["clickhouse"]["database"] = os.getenv(
        "CLICKHOUSE_DATABASE", config["database"]["clickhouse"]["database"]
    )
    config["database"]["clickhouse"]["user"] = os.getenv(
        "CLICKHOUSE_USER", config["database"]["clickhouse"]["user"]
    )
    config["database"]["clickhouse"]["password"] = os.getenv(
        "CLICKHOUSE_PASSWORD", config["database"]["clickhouse"]["password"]
    )

    return config


# Default config instance
config = load_config()
