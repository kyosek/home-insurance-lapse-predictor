import xgboost as xgb


def trainXgboost(dtrain, dtest, params):

    """Train Xgboost model using the best parameters"""

    ROUNDS = 500
    EVAL_LIST = [(dtrain, "train"), (dtest, "test")]

    xgb_model = xgb.train(params, dtrain, ROUNDS, EVAL_LIST)

    return xgb_model
