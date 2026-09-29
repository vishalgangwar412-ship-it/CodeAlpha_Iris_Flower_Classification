"""
CodeAlpha Data Science Internship
Task 1: Iris Flower Classification

This project trains a machine-learning classifier to identify
Iris flower species from sepal and petal measurements.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def main():
    # 1. Load the Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Add readable species names
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    print("First 5 rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["species"].value_counts())

    # 2. Prepare features and target
    X = df[iris.feature_names]
    y = df["target"]

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 4. Build a classification pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ])

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Make predictions
    y_pred = model.predict(X_test)

    # 7. Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    ))

    # 8. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )
    disp.plot()
    plt.title("Iris Flower Classification - Confusion Matrix")
    plt.tight_layout()

    output_path = "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\nConfusion matrix saved as: {output_path}")

    # 9. Example prediction
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)[0]
    print("\nExample flower measurements:", sample[0])
    print("Predicted species:", iris.target_names[prediction])


if __name__ == "__main__":
    main()
