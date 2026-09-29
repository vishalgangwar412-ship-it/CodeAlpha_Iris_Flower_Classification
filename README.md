# CodeAlpha Project - Iris Flower Classification

## Internship Task 1

This project is completed according to the CodeAlpha Data Science Internship Task 1 requirements.

### Task Objective

Build a machine-learning classification model that predicts the species of an Iris flower using its measurements.

The three Iris species are:

- Setosa
- Versicolor
- Virginica

### Features Used

The model uses these four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

### Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

### Machine Learning Workflow

1. Load the Iris dataset.
2. Inspect the data.
3. Separate features and target.
4. Split the data into training and testing sets.
5. Scale the feature values.
6. Train a K-Nearest Neighbors (KNN) classification model.
7. Predict flower species on test data.
8. Evaluate the model using:
   - Accuracy
   - Classification report
   - Confusion matrix
9. Display an example prediction.

### How to Run

#### 1. Install Python

Python 3.9 or newer is recommended.

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the project

```bash
python iris_classification.py
```

A confusion matrix image named `confusion_matrix.png` will also be generated.

### Dataset

The project uses Scikit-learn's built-in Iris dataset. This keeps the project self-contained and avoids requiring a separate CSV download.

### Expected Output

The program prints:

- Dataset preview
- Dataset shape
- Class distribution
- Test accuracy
- Classification report
- Confusion matrix
- Example prediction

### Project Structure

```text
CodeAlpha_Iris_Flower_Classification/
│
├── iris_classification.py
├── requirements.txt
├── README.md
├── .gitignore
└── confusion_matrix.png
```

### Internship Requirement Mapping

| CodeAlpha Task 1 Requirement | Project Implementation |
|---|---|
| Use Iris flower measurements | Four sepal/petal measurements |
| Classify Setosa, Versicolor, Virginica | KNN classifier |
| Use Scikit-learn | Scikit-learn dataset and ML tools |
| Evaluate accuracy/performance | Accuracy + classification report + confusion matrix |
| Understand classification | Complete supervised classification workflow |

## Author

Vishal Gangwar

## CodeAlpha Data Science Internship

Task 1 - Iris Flower Classification
