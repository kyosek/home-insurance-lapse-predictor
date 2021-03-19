import argparse
import logging
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# from modules.train.trainXgboost import trainXgboost
# from modules.train.tuning import hyperparameterTuning
from modules.transform.prepareInputs import prepareInput
from modules.utils.splitData import splitFeatsTraget

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s --- %(levelname)s --- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(train_required=False):
    """Main function

    Argument:
        It takes boolean for train argument.
        If True was passed, it will train the model
        If Flase, it will use the already trained model.
        Both ways, it will return the accuracy metrics at the end.

    Process:
    1. Load the datasets
    2. Data split
    3. Prepare the input
    4a. train_required = False: Load the model
    4b. train_required = True:
        Tune hyperparameters, train the model
        and save the model in "resouces/models/"
    5. Evaluate the prediction accuracy
    6. Complete the process

    """

    logging.info("Loading dataset")
    df = pd.read_csv("resources/data/home_insurance.csv")

    logging.info("Split the data")
    train, test = train_test_split(df, test_size = .3, random_state = 42)

    logging.info("Prepare the input")
    X_train, y_train = splitFeatsTraget(train)
    X_test, y_test = splitFeatsTraget(test)
    X_train, X_test = prepareInput(X_train), prepareInput(X_test)

    dtrain, dtest = (
        xgb.DMatrix(X_train, y_train, feature_names=FEATS),
        xgb.DMatrix(X_test, y_test, feature_names=FEATS),
    )

    if train_required == "False":
        logging.info("Load the model")
        xgb_model = xgb.Booster(
            model_file="resources/models/xgb_model-2021-02-07.pkl"
        )

    elif train_required == "True":
        logging.info("Start hyperparameter tuning")
        params = hyperparameterTuning(X_train, y_train, 500)

        logging.info("Start training the model")
        xgb_model = trainXgboost(dtrain, dtest, params)

        logging.info("Saving the model")
        xgb_model.save_model(
            "resources/models/xgb_model-"
            + str(datetime.today().strftime("%Y-%m-%d"))
            + ".pkl"
            )

        logging.info("Saving the feature importance list")
        pd.DataFrame.from_dict(
            xgb_model.get_score(importance_type="gain"), orient="index"
        ).sort_values(0, ascending=False).to_csv(
            "resources/feature-importance/list-"
            + str(datetime.today().strftime("%Y-%m-%d"))
            + ".csv"
        )
        
        plot = xgb.plot_importance(xgb_model.get_score(importance_type="gain"))
        plot.figure.tight_layout()
        plot.figure.savefig("resources/feature-importance/img-"
                            + str(datetime.today().strftime("%Y-%m-%d"))
                            + ".png")
        
    else:
        logging.error("Please enter the argument 'train_required' either True or False")
        exit()

    logging.info("Make predictions")
    xgb_predictions = xgb_model.predict(dtest)

    logging.info(
        "In average, there are "
        + str(
            round(
                mean_absolute_error(
                    test["prep_time_seconds"] / 60, np.expm1(xgb_predictions) / 60
                ),
                0,
            )
        )
        + " mins error"
    )

    accuracy_test = test
    accuracy_test["prediction"] = xgb_predictions

    logging.info(
        "If we exclude those orders which took more than 25 mins, there are "
        + str(round(mean_absolute_error(
                    accuracy_test[accuracy_test["prep_time_seconds"] / 60 <= 25]["prep_time_seconds"]/ 60,
                    np.expm1(accuracy_test[accuracy_test["prep_time_seconds"] / 60 <= 25]["prediction"] )/ 60,
                ), 0,)
              )
        + " mins error in average"
    )

    logging.info(
        "If we exclude those orders which took more than an hour, there are "
        + str(round(mean_absolute_error(
                    accuracy_test[accuracy_test["prep_time_seconds"] / 60 <= 60][
                        "prep_time_seconds"]/ 60,
                    np.expm1(accuracy_test[accuracy_test["prep_time_seconds"] / 60 <= 60][
                            "prediction"])/ 60,
                ),0,)
            )
        + " mins error in average"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Deliveroo - algos-takehome")

    parser.add_argument("train_required", type=str, help="wether want to train")

    args = parser.parse_args()

    main(args.train_required)
