#model training and feature visualization 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def plot_distributions(df_pd):
    # PlayTime Hours Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df_pd["PlayTimeHours"], kde=True, bins=30, color='skyblue')
    plt.title("PlayTime Hours Distribution")
    plt.xlabel("PlayTime Hours")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Player Level Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df_pd["PlayerLevel"], kde=True, bins=30, color='orange')
    plt.title("Player Level Distribution")
    plt.xlabel("Player Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Achievements Unlocked Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df_pd["AchievementsUnlocked"], kde=True, bins=30, color='green')
    plt.title("Achievements Unlocked Distribution")
    plt.xlabel("Achievements Unlocked")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Engagement Score Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df_pd["EngagementScore"], kde=True, bins=30, color='purple')
    plt.title("Engagement Score Distribution")
    plt.xlabel("Engagement Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    
def visualize_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

def train_model(df, features):
    df_pd = df.toPandas()

    # Visualizations
    plot_distributions(df_pd)
    # Model Training
    X = df_pd[features]
    y = df_pd["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    visualize_confusion_matrix(y_test, y_pred)

    return clf, df_pd


