# authors: Luke Yang, HanChen Wang
# date: 2022-12-10

"""This script reads the preprocessed train and test data files and perform comparative analysis on them for different ML models. It further selects Logostic Regression Classifier as the best model and performs hyperparameter optimizaiton. The final results are saves as .csv and .png files in the output folder. 

Usage: data_analysis.py --traindata=<traindata> --testdata=<testdata> --output=<output> ...

Arguments:
  --traindata=<traindata>        path of the training data
  --testdata=<testdata>         path of the testing data
  --output=<output>     folder that stores the generated plots

Make sure you call this script in the repo's root path
Example:
python src/data_analysis.py --traindata=data/processed/train_cleaned.csv --testdata=data/processed/test_cleaned.csv --output=results/model

"""

# Imports
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hashlib import sha1
from docopt import docopt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import cross_validate

# Code from Varada, DSCI 573, UBC Master of Data Science course
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def perform_ml_analysis(train_data, test_data, out_path):
    # perform ml analysis for different models and pick best model
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_df = pd.read_csv(train_data, index_col=0, encoding="utf-8")
    test_df = pd.read_csv(test_data, index_col=0, encoding="utf-8")

    # 1. Create the column transformer / preprocessor
    # For the numeric features, we will Standardize the numbers.
    # For the categorical features, we will apply OneHotEncoder on them to convert
    # the column into multiple binary features.
    # Many other features already have good encoding, so we don't
    # have to modify them (passthrough).

    numeric_features = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
        "days_from_last_review",
    ]
    categorical_features = [
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
    ]
    discretization_features = ["latitude", "longitude"]
    drop = ["id", "last_review", "host_id", "host_name", "name"]

    preprocessor = make_column_transformer(
        (
            make_pipeline(StandardScaler(), SimpleImputer(strategy="most_frequent")),
            numeric_features,
        ),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        (KBinsDiscretizer(n_bins=30, encode="onehot"), discretization_features),
        ("drop", drop),
    )

    preprocessor

    # Show the preprocessor
    print(
        """
    ################################################
    #        Column_transformer                    #
    ################################################
    """
    )
    print(preprocessor)

    # 2. Fit and transform on the training data
    X_train, y_train = (
        train_df.drop(columns=["reviews_per_month"]),
        train_df["reviews_per_month"],
    )
    X_test, y_test = (
        test_df.drop(columns=["reviews_per_month"]),
        test_df["reviews_per_month"],
    )

    # This line nicely formats the feature names from `preprocessor.get_feature_names_out()`
    # so that we can more easily use them below
    preprocessor.verbose_feature_names_out = False
    # Create a dataframe with the transformed features and column names
    preprocessor.fit(X_train)

    # transformed data
    X_train_transformed = preprocessor.transform(X_train)
    ohe_features = (
        preprocessor.named_transformers_["onehotencoder"]
        .get_feature_names_out()
        .tolist()
    )

    discretization_features = (
        preprocessor.named_transformers_["kbinsdiscretizer"]
        .get_feature_names_out()
        .tolist()
    )

    # Code to get all the feature names
    feature_names = numeric_features + ohe_features + discretization_features

    X_train_enc = pd.DataFrame(X_train_transformed, columns=feature_names)

    # Show the transformed data
    print(
        """
    ################################################
    #        X_train_transformed                   #
    ################################################
    """
    )
    print(X_train_enc)

    models = {
        "Dummy": DummyRegressor(),
        "Ridge": Ridge(),
        "Random Forests": RandomForestRegressor(n_jobs=-1, random_state=573),
        "SVR": SVR(),
        "LGBMR": LGBMRegressor(random_state=573),
        "CatBoostRegressor": CatBoostRegressor(verbose=False, random_state=573),
    }

    cross_val_results = {}
    for model in models:
        print('Analyzing', model)
        cross_val_results[model] = mean_std_cross_val_scores(
            make_pipeline(preprocessor, models[model]),
            X_train,
            y_train,
            cv=5,
            return_train_score=True,
        )

    from sklearn.feature_selection import RFECV

    cross_val_results["CatBoost_rfe"] = mean_std_cross_val_scores(
        make_pipeline(
            preprocessor,
            RFECV(Ridge(), cv=10),
            CatBoostRegressor(verbose=False, random_state=573),
        ),
        X_train,
        y_train,
        return_train_score=True,
    )
    cross_val_results_df = pd.DataFrame(cross_val_results)

    # Show cross validation results for different modesl
    print(
        """
    ################################################
    #      CV Results from different models        #
    ################################################
    """
    )
    print(cross_val_results_df)
    cross_val_results_df.to_csv(out_path + "model_selection.csv")

    # We select CatBoostRegressor for model hyperparameter optimization.

    lr = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

    import numpy as np

    param_grid = {
        "catboostregressor__learning_rate": np.arange(0.01, 0.1, 0.02),
        "catboostregressor__max_depth": np.arange(4, 10, 1),
    }

    from sklearn.model_selection import GridSearchCV

    pipe_cat = make_pipeline(
        preprocessor, CatBoostRegressor(verbose=False, random_state=573)
    )

    grid_search = GridSearchCV(
        pipe_cat,
        param_grid,
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    # Best estimator from random search
    print(
        """
    ################################################
    #            Best estimator                    #
    ################################################
    """
    )
    print(grid_search.best_estimator_)

    # This is an improvement compared to the mean validation f1 score using default parameters.
    print(
        """
    ################################################
    #      R2 score of the best estimator          #
    ################################################
    """
    )
    print(grid_search.best_score_)

    # This is the optimized hyperparameter.
    print(
        """
    ################################################
    #      parameters of the best estimator        #
    ################################################
    """
    )
    print(grid_search.best_params_)

    # Generate a table showing feature importances.
    feat_importance = grid_search.best_estimator_.named_steps[
        "catboostregressor"
    ].feature_importances_

    # Code to get all the feature names
    feature_names = numeric_features + ohe_features + discretization_features

    # This is the feature importances of the optimized model.
    print(
        """
    ################################################
    #   Feature importances of the best estimator  #
    ################################################
    """
    )
    feature_table = pd.DataFrame(
        {"Feature": feature_names, "Feature Importance": feat_importance}
    ).sort_values(by="Feature Importance", ascending=False)
    print(feature_table)
    feature_table.to_csv(out_path + "feature_Importance.csv")

    # Finally, check the f1_score of the test data with our optimized model.
    test_score = grid_search.score(X_test, y_test)
    print(f"The R2_score of the test data is", round(test_score, 3))


if __name__ == "__main__":
    arguments = docopt(__doc__)

    train_data = arguments["--traindata"]  # load 1 dataset at a time
    test_data = arguments["--testdata"]  # load 1 dataset at a time
    out_path = arguments["--output"][0]

    assert train_data.endswith(".csv")
    assert test_data.endswith(".csv")
    assert "results" in out_path

    perform_ml_analysis(train_data, test_data, out_path)

    # Tests that the files are generated as expected
    assert os.path.exists("./results/model/feature_Importance.csv")
    assert os.path.exists("./results/model/model_selection.csv")
