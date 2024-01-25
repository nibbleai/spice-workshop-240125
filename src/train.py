import logging

from src.data import get_train_dataset, get_target, train_test_split
from src.preprocessing import preprocess
from src.model import get_model, evaluate
from src.schemas import TaxiColumn

logger = logging.getLogger(__name__)


def main():
    data = get_train_dataset()
    target = get_target(data)
    data, target = preprocess(data, target)

    train_data, test_data = train_test_split(data)
    train_target = target.loc[train_data.index]
    test_target = target.loc[test_data.index]

    feature_columns = [
        TaxiColumn.PICKUP_LAT,
        TaxiColumn.PICKUP_LON,
        TaxiColumn.DROPOFF_LAT,
        TaxiColumn.DROPOFF_LON,
    ]
    train_features = train_data.loc[:, feature_columns]
    test_features = test_data.loc[:, feature_columns]

    logger.info("Training model...")
    model = get_model().fit(train_features, train_target)

    metrics = evaluate(model, features=test_features, target=test_target)
    logger.info(f"Model metrics are:\n{metrics}")


if __name__ == '__main__':
    main()
