import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initial Setup + Spark Configuration
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

file_path = r"C:\Users\Dell\Desktop\HIDE\gaming_data.csv"
print(f"File exists: {os.path.isfile(file_path)}")

spark = (
    SparkSession.builder
    .appName("GamingBehaviorPrediction")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(file_path, header=True, inferSchema=True)

# EDA + Feature Engineering
df.select("PlayTimeHours").summary("min", "max", "count").show()
zero_playtime_count = df.filter(col("PlayTimeHours") == 0).count()
print(f"Number of players with zero playtime: {zero_playtime_count}")

df = df.withColumn("churn", when(col("PlayTimeHours") < 1, 1).otherwise(0))
df = df.withColumn("is_male", when(col("Gender") == "Male", 1).otherwise(0))

df = df.withColumn("PlaytimePerLevel", when(col("PlayerLevel") > 0, col("PlayTimeHours") / col("PlayerLevel")).otherwise(0))
df = df.withColumn("AchievementsPerLevel", when(col("PlayerLevel") > 0, col("AchievementsUnlocked") / col("PlayerLevel")).otherwise(0))
df = df.withColumn("PlaytimePurchasesRatio", when(col("InGamePurchases") > 0, col("PlayTimeHours") / col("InGamePurchases")).otherwise(0))

df = df.withColumn("EngagementScore", 
                   col("PlayTimeHours") * 0.3 + 
                   col("InGamePurchases") * 0.2 + 
                   col("AchievementsUnlocked") * 0.2 + 
                   (col("PlayerLevel") / 10) * 0.3)

df = df.withColumn("EngagementLevelNum",
                   when(col("EngagementScore") >= 20, 2)
                   .when(col("EngagementScore") >= 10, 1)
                   .otherwise(0))

# Prepare Data + Train/Test Split
features = ["PlayTimeHours", "InGamePurchases", "PlayerLevel", "is_male", "AchievementsUnlocked",
            "PlaytimePerLevel", "AchievementsPerLevel", "PlaytimePurchasesRatio",
            "EngagementScore", "EngagementLevelNum"]

df = df.select(*features, "churn").na.drop()
df_pd = df.toPandas()

print("\nChurn value counts:")
churn_counts = df_pd['churn'].value_counts()
print(churn_counts)

# Get churn counts
labels = ['Active (0)', 'Churned (1)']
colors = ['yellow', 'blue']

# Create pie chart with percentage and count
plt.figure(figsize=(6, 6))
plt.pie(
    churn_counts,
    labels=[f"{label}: {count} ({count / sum(churn_counts) * 100:.1f}%)"
            for label, count in zip(labels, churn_counts)],
    colors=colors,
    startangle=90,
    counterclock=False
)
plt.title("Churn Distribution")
plt.tight_layout()
plt.show()

X = df_pd[features]
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training + Evaluation
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

spark.stop()