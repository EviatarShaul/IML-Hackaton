# Task 1.2.3
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from project.task_1.code.hackathon_code.utils.preprocess import create_x_y_df

LABEL_NAME = "cancellation_datetime"
DATAFRAME_IMPORTANT_COLS = ["guest_is_not_the_customer",
                            "no_of_adults",
                            "no_of_children",
                            "no_of_extra_bed",
                            "no_of_room",
                            "cost",
                            "time_from_booking_to_checkin",
                            "stay_duration",
                            "is_weekday",
                            "checkin_month_sin",
                            "checkin_month_cos",
                            "hotel_age",
                            "hotel_star_rating",
                            "special_requests"]


def person_corr(X: pd.DataFrame, y: np.ndarray) -> list:
    corr_list = []
    # calculating correlation for each header
    for name in X.columns:
        coef = np.corrcoef(X[name].values, y)[0, 1]
        print(f"{name}:\t{coef}")
        corr_list.append(coef)
    return corr_list


def churn_prediction_model(raw_data: pd.DataFrame):
    X, y = create_x_y_df(raw_data, DATAFRAME_IMPORTANT_COLS, LABEL_NAME)

    # TODO: this part should be done in pre-process!
    X["special_requests"].replace(np.nan, 0, inplace=True)
    y = np.where(pd.Series(y).isnull(), 0, 1)

    # calculating person correlation:
    corr_list = person_corr(X, y)
    # print(corr_list, "\n")

    # calculating pca:
    pca = PCA(n_components=5)
    pca.fit(X, y)
    # print(pca.explained_variance_)
    # print(pca.components_)




