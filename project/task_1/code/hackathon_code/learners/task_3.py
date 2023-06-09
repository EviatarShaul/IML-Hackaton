# Task 1.2.3
import joblib
import sys

from plotly.subplots import make_subplots

sys.path.append("../")
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from project.task_1.code.hackathon_code.utils.preprocess import create_x_y_df
from task_1.code.hackathon_code.utils.model_helper import load_model

MODEL_LOAD_PATH = "task_1_model_weights.sav"
LABEL_NAME = "cancellation_datetime"
DATAFRAME_IMPORTANT_COLS = ["guest_is_not_the_customer",
                            "no_of_adults",
                            "no_of_children",
                            "no_of_extra_bed",
                            "no_of_room",
                            "time_from_booking_to_checkin",
                            "stay_duration",
                            "is_weekday",
                            "checkin_month_sin",
                            "checkin_month_cos",
                            "hotel_age",
                            "hotel_star_rating",
                            "special_requests"]
CATEGORICAL_COLS = ["charge_option_", "accommadation_type_name", "original_payment_type_"]


def person_corr(X: pd.DataFrame, y: np.ndarray) -> list:
    corr_list = []
    # calculating correlation for each header
    for name in X.columns:
        coef = np.corrcoef(X[name].values, y)[0, 1]
        corr_list.append(coef)
    return corr_list


def churn_prediction_model(raw_data: pd.DataFrame):
    # for col in raw_data.columns:
    #     for prefix in CATEGORICAL_COLS:
    #         if col.startswith(prefix):
    #             DATAFRAME_IMPORTANT_COLS.append(col)

    # Splitting to X and y
    X, y = create_x_y_df(raw_data, DATAFRAME_IMPORTANT_COLS, LABEL_NAME)
    y = np.where(pd.Series(y).isnull(), 0, 1)

    # calculating person correlation:
    corr_list = person_corr(X, y)
    corr_list = np.array(corr_list)
    corr_sorted_ind = np.argsort(corr_list)
    print(corr_list[corr_sorted_ind], "\n")
    print(np.array(X.columns[corr_sorted_ind]), "\n")

    # calculating pca:
    pca = PCA(n_components=5)
    pca.fit(X, y)
    # print(pca.explained_variance_)
    # print(pca.components_)

    # Calculating
    cols_high_corr = []
    for name1 in X.columns:
        for name2 in X.columns:
            if name1 in cols_high_corr or name2 in cols_high_corr:
                continue
            if name1 != name2:
                cor = np.corrcoef(X[name1].values, X[name2].values)[0, 1]
                if np.abs(cor) > 0.5:
                    print(name1, name2, cor)
                    cols_high_corr.append(name2)
    print(cols_high_corr)
    # X.drop(cols_high_corr)

    # Displaying the lambdas:
    lambdas = 10 ** np.linspace(-3, 2, 100)
    models = [dict(name="Lasso",
                   model=lambda lam, x, y: Lasso(alpha=lam, normalize=True,
                                                 max_iter=10000, tol=1e-4).fit(
                       x, y),
                   reg_penalty=lambda lam, w: lam * np.linalg.norm(w, ord=1)),
              dict(name="Ridge",
                   model=lambda lam, x, y: Ridge(alpha=lam,
                                                 normalize=True).fit(x, y),
                   reg_penalty=lambda lam, w: lam * np.linalg.norm(w, ord=2))]
    regressors = {}
    for m in models:
        res = dict(coefs=pd.DataFrame([], columns=list(X.columns),
                                      index=lambdas),
                   losses=pd.DataFrame([], columns=["mse", "reg", "loss"],
                                       index=lambdas))

        for lam in lambdas:
            model = m["model"](lam, X, y)
            res["coefs"].loc[lam, :] = model.coef_

            mse = mean_squared_error(y, model.predict(X))
            reg = m["reg_penalty"](lam, model.coef_)
            res["losses"].loc[lam, :] = [mse, reg, mse + reg]

        regressors[m["name"]] = res

    coefs, losses = regressors["Ridge"].values()

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[r'$\text{Regularization Path}$',
                                        r'$\text{Model Losses}$'],
                        row_heights=[400, 200], vertical_spacing=.1)

    # Plot the regularization path for each feature
    for i, col in enumerate(X.columns):
        fig.add_trace(
            go.Scatter(x=lambdas, y=coefs.loc[:, col], mode='lines', name=col,
                       legendgroup="1"))

    # Plot the losses graph and mark lambda with lowest loss
    lam = np.argmin(losses.loc[:, 'loss'])
    fig.add_traces([go.Scatter(x=lambdas, y=losses.loc[:, 'mse'], mode='lines',
                               name="Fidelity Term - MSE", legendgroup="2"),
                    go.Scatter(x=lambdas, y=losses.loc[:, 'reg'], mode='lines',
                               name="Regularization Term", legendgroup="2"),
                    go.Scatter(x=lambdas, y=losses.loc[:, 'loss'],
                               mode='lines', name="Joint Loss",
                               legendgroup="2"),
                    go.Scatter(x=[lambdas[lam]],
                               y=[losses.loc[:, 'loss'].values[lam]],
                               mode='markers', showlegend=False,
                               marker=dict(size=8, symbol="x"),
                               hovertemplate="Lambda: %{x}<extra></extra>")],
                   2, 1)

    fig.update_layout(hovermode='x unified', margin=dict(t=50),
                      legend=dict(tracegroupgap=60),
                      title=r"$(1)\text{ Fitting Ridge Regression}$") \
        .update_xaxes(type="log").show()

    r"""
    model = load_model(MODEL_LOAD_PATH)
    y_prob = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         # marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    """




