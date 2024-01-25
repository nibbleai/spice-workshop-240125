import logging

import pandas as pd

from src.data import get_live_dataset
from src.features import base, spatial, temporal, weather
from src.io_ import load_generator, load_model
from src.resources import get_resources
from src.schemas import TaxiColumn

logger = logging.getLogger(__name__)


def main():
    live_data = get_live_dataset()

    logger.info("Loading features generator...")
    features_generator = load_generator()
    features_generator.resources = get_resources()

    live_features = (
        features_generator
        .transform(live_data, tags={'dataset': 'live'})
        .to_pandas()
    )

    logger.info("Loading learner...")
    learner = load_model()

    logger.info("Making predictions...")
    predicted = pd.Series(learner.predict(live_features), name="duration")

    trips_with_prediction = pd.concat([live_data, predicted], axis=1)
    logger.info(
        f"Predicted {len(predicted)} trips, spanning from "
        f"{trips_with_prediction[TaxiColumn.PICKUP_TIME].dt.date.min()} to "
        f"{trips_with_prediction[TaxiColumn.PICKUP_TIME].dt.date.max()}:\n"
        f"{trips_with_prediction}"
    )


if __name__ == '__main__':
    main()
