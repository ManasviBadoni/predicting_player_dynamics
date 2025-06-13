import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

# Set PySpark environment
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

# Start Spark session
spark = (
    SparkSession.builder
    .appName("GamingBehaviorPrediction")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

def load_and_process_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Feature Engineering
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

    # Final feature list
    features = [
        "PlayTimeHours", "InGamePurchases", "PlayerLevel", "is_male", "AchievementsUnlocked",
        "PlaytimePerLevel", "AchievementsPerLevel", "PlaytimePurchasesRatio",
        "EngagementScore", "EngagementLevelNum"
    ]
    
    # Drop nulls and select relevant features
    df = df.select(*features, "churn").na.drop()

    # Convert churn column for visualization
    df_pd = df.select("churn").toPandas()

    # Churn Class Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="churn", data=df_pd, palette=["skyblue", "salmon"])
    plt.title("Churn Class Distribution")
    plt.xlabel("Churn Class (0 = Stay, 1 = Churn)")
    plt.ylabel("Number of Players")
    plt.xticks([0, 1], ["Stay", "Churn"])
    plt.tight_layout()
    plt.show()

    return df, features
