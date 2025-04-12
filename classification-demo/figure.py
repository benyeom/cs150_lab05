import numpy as np
import plotly.graph_objects as go
from dash import html
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def create_contour_figure(sample_size=100, threshold=0.5):
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if sample_size < 100:
        frac = sample_size / 100.0
        indices = np.random.choice(len(X_train), size=int(len(X_train) * frac), replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = LogisticRegression(random_state=0)
    model.fit(X_train_std, y_train)

    train_probs = model.predict_proba(X_train_std)[:, 1]
    test_probs = model.predict_proba(X_test_std)[:, 1]
    y_train_pred = (train_probs >= threshold).astype(int)
    y_test_pred = (test_probs >= threshold).astype(int)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    x_min, x_max = X_train_std[:, 0].min() - 0.5, X_train_std[:, 0].max() + 0.5
    y_min, y_max = X_train_std[:, 1].min() - 0.5, X_train_std[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid)[:, 1]
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        colorscale=[
            (0, "rgb(198,225,255)"),
            (threshold, "rgb(198,225,255)"),
            (threshold, "rgb(255,198,198)"),
            (1, "rgb(255,198,198)")
        ],
        zmin=0,
        zmax=1,
        contours_coloring='fill',
        line_smoothing=1.3,
        showscale=False
    ))

    X_train_0 = X_train_std[y_train == 0]
    X_train_1 = X_train_std[y_train == 1]
    fig.add_trace(go.Scatter(
        x=X_train_0[:, 0],
        y=X_train_0[:, 1],
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='blue',
        marker_size=7,
        name=f"Training Data (acc={train_acc:.3f})",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=X_train_1[:, 0],
        y=X_train_1[:, 1],
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='red',
        marker_size=7,
        name=f"Training Data (acc={train_acc:.3f})",
        showlegend=False
    ))

    X_test_0 = X_test_std[y_test == 0]
    X_test_1 = X_test_std[y_test == 1]
    fig.add_trace(go.Scatter(
        x=X_test_0[:, 0],
        y=X_test_0[:, 1],
        mode='markers',
        marker_symbol='circle',
        marker_color='blue',
        marker_size=7,
        name=f"Test Data (acc={test_acc:.3f})",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=X_test_1[:, 0],
        y=X_test_1[:, 1],
        mode='markers',
        marker_symbol='circle',
        marker_color='red',
        marker_size=7,
        name=f"Test Data (acc={test_acc:.3f})",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='orange', size=7),
        name=f"Threshold ({threshold:.3f})"
    ))

    return fig


def create_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name="ROC Curve"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random", line=dict(dash='dash')))
    fig.update_layout(
        title=f"ROC Curve (AUC = {auc_val:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig


def create_confusion_matrix_table(y_test, y_pred_class):
    cm = confusion_matrix(y_test, y_pred_class)
    table = html.Div([
        html.H3("Confusion Matrix", style={'textAlign': 'center', 'marginBottom': '10px'}),
        html.Table([
            html.Thead(
                html.Tr([html.Th(""), html.Th("Predicted 0"), html.Th("Predicted 1")])
            ),
            html.Tbody([
                html.Tr([html.Td("Actual 0"), html.Td(cm[0, 0]), html.Td(cm[0, 1])]),
                html.Tr([html.Td("Actual 1"), html.Td(cm[1, 0]), html.Td(cm[1, 1])]),
            ])
        ], style={
            'width': '100%',
            'border': '2px solid black',
            'textAlign': 'center',
            'fontSize': '18px',
            'padding': '10px'
        })
    ], style={'width': '70%', 'margin': 'auto'})
    return table

