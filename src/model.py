from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import config

__all__ = ['get_model', 'evaluate']


def get_model():
    return RandomForestRegressor()


def cross_validate(*, generator, learner, data, target):
    splitter = TimeSeriesSplit(config.cv_splits)

    return [
        _validate(
            generator=generator,
            learner=learner,
            X_train=data.iloc[train_idx],
            y_train=target.iloc[train_idx],
            X_test=data.iloc[test_idx],
            y_test=target.iloc[test_idx],
            tags={'fold': i}
        )
        for i, (train_idx, test_idx) in enumerate(splitter.split(data, target))
    ]


def train(*, generator, learner, X, y, tags):
    fitted_generator = generator.fit(X, tags=tags)
    Xt = fitted_generator.transform(X, tags=tags).to_pandas()
    return fitted_generator, learner.fit(Xt, y)


def _validate(*, generator, learner, X_train, y_train, X_test, y_test, tags):
    generator, learner = train(
        generator=generator,
        learner=learner,
        X=X_train,
        y=y_train,
        tags={**tags, 'step': 'train'}
    )

    return _evaluate(
        generator=generator,
        learner=learner,
        X=X_test,
        y=y_test,
        tags={**tags, 'step': 'test'}
    )


def _evaluate(*, generator, learner, X, y, tags):
    Xt = generator.transform(X, tags=tags).to_pandas()
    return mean_squared_log_error(y, learner.predict(Xt))


