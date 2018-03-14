import numpy as np


# Absolute Error
def ae(test, predict):
    return abs(test - predict)


# Mean Absolute Error
def mae(test, predict):
    return ae(test, predict).mean()


# Squared Error
def se(test, predict):
    return (test - predict) ** 2


# Mean Squared Error
def mse(test, predict):
    return se(test, predict).mean()


# Root Mean Squared Error
def rmse(test, predict):
    return np.sqrt(mse(test, predict))
