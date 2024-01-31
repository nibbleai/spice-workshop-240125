from spice import Registry

registry = Registry("taxi")


@registry.register(depends=["pickup_time"])
def pickup_hour(pickup_time):
    return pickup_time.dt.hour


@registry.register(depends=["pickup_time"])
def pickup_weekday(pickup_time):
    return pickup_time.dt.weekday


@registry.register(name="pickup_time")
def pickup_time(data):
    return data["pickup_datetime"]
