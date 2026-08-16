

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.database import initialize_timescaledb, save_measurements
from src.generate_dataset import resolve_data_path
from src.preprocessing import to_database_frame


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="engineered_air_weather.csv",
        help="CSV path relative to data/",
    )
    arguments = parser.parse_args()
    input_path = resolve_data_path(arguments.input)
    if not input_path.is_file():
        raise SystemExit("Input dataset was not found under data/")

    engineered = pd.read_csv(input_path)
    database_frame = to_database_frame(engineered)
    initialize_timescaledb()
    inserted = save_measurements(database_frame)
    LOGGER.info("Upserted %d measurements into TimescaleDB", inserted)
    print(f"TimescaleDB upserted rows: {inserted}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
