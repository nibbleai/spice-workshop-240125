import logging

import numpy as np
from spice import Generator

from src.data import get_train_dataset, get_target
from src.features.registry import registry
from src.features import base, spatial, temporal, weather
from src.io_ import save_generator, save_model
from src.preprocessing import preprocess
from src.model import cross_validate, get_model, train
from src.resources import get_resources
from src.schemas import TaxiColumn

logger = logging.getLogger(__name__)


def main():
    data = get_train_dataset().sort_values(TaxiColumn.PICKUP_TIME)
    target = get_target(data)
    data, target = preprocess(data, target)

    features_generator = Generator(
        registry,
        resources=get_resources(),
        features=[
            "cyclical_pickup_hour",
            "quantile_bin_hour",
            "is_raining",
            "is_rush_hour",
            "euclidean_distance",
            "manhattan_distance",
            "euclidean_manhattan_ratio",
            "pca_pickup",
            "pca_dropoff",
            "pickup_cluster",
            "dropoff_cluster"
        ]
    )

    learner = get_model()
    metrics = cross_validate(
        generator=features_generator,
        learner=learner,
        data=data,
        target=target
    )
    logger.info(
        "Model metrics are: "
        f"mean={np.mean(metrics)} "
        f"std={np.std(metrics)}"
    )

    logger.info(f"Training model...")
    features_generator, learner = train(
        generator=features_generator,
        learner=learner,
        X=data,
        y=target,
        tags={'dataset': 'train'}
    )

    logger.info("Saving generator...")
    save_generator(features_generator)

    logger.info("Saving model...")
    save_model(learner)


if __name__ == '__main__':
    main()
