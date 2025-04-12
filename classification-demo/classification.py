import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def run_classification(sample_fraction=1.0, threshold=0.5):
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    #split dataset: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=2
    )

    if sample_fraction < 1.0:
        X_train = X_train.sample(frac=sample_fraction, random_state=2)
        y_train = y_train.loc[X_train.index]

    #build and train the model using top 50 words.
    model = Pipeline([
        ('vectorizer', CountVectorizer(max_features=50)),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)

    #get continuous predictions, threshold them, and clip for ROC calculations.
    y_pred_cont = model.predict(X_test)
    y_pred_class = (y_pred_cont >= threshold).astype(int)
    y_prob = np.clip(y_pred_cont, 0, 1)

    return y_test, y_pred_class, y_prob, model


if __name__ == "__main__":
    #for testing only: run with default parameters.
    y_test, y_pred_class, y_prob, _ = run_classification()
    from sklearn.metrics import accuracy_score, roc_auc_score

    accuracy = accuracy_score(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_prob)
    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)
