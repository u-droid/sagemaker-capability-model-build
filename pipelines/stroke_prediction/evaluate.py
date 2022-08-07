"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)
    logger.info(f"Prediction: {predictions[:20]}")
    logger.debug("Calculating Metric.")
    metric_tup = precision_recall_fscore_support(y_test, np.round(abs(predictions)), average='weighted')
    report_dict = {
        "classification_metrics": {
            "precision": {
                "value": metric_tup[0]
            },
            "recall":{
                "value": metric_tup[1]
            },
            "fscore":{
                "value": metric_tup[2]
            }
        }
    }
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing out evaluation report : {report_dict}")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict, cls=NpEncoder))
