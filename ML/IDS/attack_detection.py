
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

benign_train = pd.read_csv(r"data\benign_training_data.csv")
attack_train = pd.read_csv(r"data\attack_training_data.csv")
train_df = pd.concat([benign_train, attack_train], ignore_index=True)

test_benign = pd.read_csv(r"data\benign_test_data.csv")
test_attack = pd.read_csv(r"data\attack_test_data.csv")
test_benign["label"] = 0
test_benign["class"] = "Benign"
test_attack["label"] = 1
test_attack["class"] = "Attack"
test_df = pd.concat([test_benign, test_attack], ignore_index=True)
y_test = test_df["label"]

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)


non_features = ['class', 'label']
X_train = train_df.drop(columns=non_features)
y_train = train_df['label']
X_test = test_df.drop(columns=non_features)

X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_params = {
    'n_neighbors': [3, 5, 7,9,11],
    'weights': ['distance'],
    'metric': ['euclidean', 'manhattan'],
    'p': [1, 2],
    'leaf_size': [20, 30, 40]
}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params, scoring='f1', cv=5)
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_
lr_params = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10],
    'class_weight': ['balanced']
}
models = {
    "Naive Bayes": GaussianNB(),
    "KNN ": best_knn,
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": GridSearchCV(LogisticRegression(max_iter=1000), lr_params, scoring='f1', cv=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, criterion='entropy', class_weight='balanced', random_state=42)
}

metrics_table = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    metrics_table.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    })

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"{name.replace(' ', '_').lower()}_predictions_pca.csv", index=False)

metrics_df = pd.DataFrame(metrics_table)
print("\n=== Model Evaluation Table  ===")
print(metrics_df.to_string(index=False))
models = metrics_df['Model'].tolist()


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Define custom colors for each model
custom_colors = ['orange','lightblue','lightgreen','gray','lightpink','peachpuff']  # (as many as models you have)
# Set figure
plt.figure(figsize=(14, 7))

# Set bar width
bar_width = 0.13
index = np.arange(len(metrics))

metrics_df_percent = metrics_df.copy()
metrics_df_percent[['Accuracy', 'Precision', 'Recall', 'F1 Score']] *= 100
bars=[]
# Plot bars for each model
for i, model in enumerate(models):
    bar=plt.bar(index + i*bar_width, metrics_df_percent.loc[i, metrics], bar_width, label=model,color=custom_colors[i])
    bars.append(bar)

plt.xticks(index + bar_width*(len(models)/2 - 0.5), metrics, rotation=0)
plt.ylim(0, 100)
plt.legend(
    title="Models",
    loc='lower center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=len(models),
    frameon=False
)
# Annotate numbers inside bars
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,   # center of bar
            height / 2,                         # halfway up inside the bar
            f'{height:.2f}%',
            ha='center', va='center', rotation=90, fontsize=10, color='white',fontweight='bold'
        )
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()