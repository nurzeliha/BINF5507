import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

# Prepare features and target
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a random forest classifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

# Predict probabilities
y_scores_forest = forest.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr_forest, tpr_forest, _ = roc_curve(y_test, y_scores_forest)
roc_auc_forest = auc(fpr_forest, tpr_forest)

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_scores_forest)
average_precision = average_precision_score(y_test, y_scores_forest)

# Calculate the minority class proportion for the random baseline
minor_class = y_test.mean()

def plot_curves(
    tpr, fpr, auroc, precision, recall, auprc, model_name, minority_class=0.1
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ROC curve
    ax1.plot(fpr, tpr, color="green", lw=2, label="AUROC = %0.2f" % auroc)
    ax1.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("{} ROC Curve".format(model_name))
    ax1.legend(loc="lower right")

    # Plot Precision-Recall curve
    ax2.plot(
        recall, precision, color="purple", lw=2, label="AUPRC = %0.2f" % auprc
    )
    ax2.axhline(
        y=minority_class, color="red", lw=2, linestyle="--", label="Random"
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("{} Precision-Recall Curve".format(model_name))
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# Plot the curves
plot_curves(
    tpr_forest,
    fpr_forest,
    roc_auc_forest,
    precision,
    recall,
    average_precision,
    "Random Forest",
    minority_class=minor_class,
)


# Predict on the test set
y_pred_forest = forest.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_forest)
precision = precision_score(y_test, y_pred_forest)
recall = recall_score(y_test, y_pred_forest)
f1 = f1_score(y_test, y_pred_forest)

print(f"Random Forest - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")