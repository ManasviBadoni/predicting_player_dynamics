from dataload import load_and_process_data, spark
from modeltrain import train_model
from userinput import predict_churn_from_input

file_path = r"C:\Users\Dell\OneDrive\Desktop\PLAYERCHURN\gaming_data.csv"

df, features = load_and_process_data(file_path)
clf, df_pd = train_model(df, features)
predict_churn_from_input(clf, features, df_pd)

spark.stop()
