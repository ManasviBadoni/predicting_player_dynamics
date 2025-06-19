import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from dataload import load_and_process_data, spark
from modeltrain import train_model
from userinput import visualize_engagement_playtime

# Load data and train model
file_path = r"C:\Users\Dell\OneDrive\Desktop\PLAYERCHURN\gaming_data.csv"
df_spark, features = load_and_process_data(file_path)
clf, df_pd = train_model(df_spark, features)

class ChurnGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicting Player Dynamics – by Data Dynamos")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Background image
        image_path = "bg.png"
        self.bg_image = Image.open(image_path).resize((800, 600))
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Title
        self.title_label = tk.Label(
            root,
            text="Predicting Player Dynamics\nby Data Dynamos",
            font=("Helvetica", 20, "bold"),
            fg="#00ffcc",
            bg="black"
        )
        self.title_label.place(relx=0.5, y=20, anchor="n")

        # Input Fields
        self.entries = {}
        self.gender_var = tk.StringVar(value="Male")
        y_base = 120
        fields = [
            ("PlayTimeHours", "Play Time (hours):"),
            ("InGamePurchases", "In-Game Purchases:"),
            ("PlayerLevel", "Player Level:"),
            ("AchievementsUnlocked", "Achievements Unlocked:")
        ]

        for i, (key, label) in enumerate(fields):
            tk.Label(root, text=label, font=("Helvetica", 11, "bold"),
                     fg="#00ffcc", bg="black").place(x=180, y=y_base + i * 40)
            entry = tk.Entry(root, font=("Helvetica", 11), width=22,
                             bg="#222222", fg="#00ffcc", insertbackground="#00ffcc", relief="flat")
            entry.place(x=400, y=y_base + i * 40)
            self.entries[key] = entry

        # Gender dropdown
        tk.Label(root, text="Gender:", font=("Helvetica", 11, "bold"),
                 fg="#00ffcc", bg="black").place(x=180, y=y_base + 160)
        gender_menu = ttk.Combobox(root, textvariable=self.gender_var,
                                   values=["Male", "Female"], state="readonly", width=20)
        gender_menu.place(x=400, y=y_base + 160)
        gender_menu.configure(font=("Helvetica", 10))

        # Predict Button
        self.predict_btn = tk.Button(root, text="Predict Churn",
                                     font=("Helvetica", 12, "bold"),
                                     bg="black", fg="#00ffcc",
                                     activebackground="#00ffcc",
                                     activeforeground="black",
                                     relief="flat",
                                     command=self.predict_churn)
        self.predict_btn.place(relx=0.5, y=y_base + 220, anchor="center")
        self.predict_btn.bind("<Enter>", lambda e: self.predict_btn.config(bg="#00ffcc", fg="black"))
        self.predict_btn.bind("<Leave>", lambda e: self.predict_btn.config(bg="black", fg="#00ffcc"))

        # Result Label
        self.result_label = tk.Label(root, text="", font=("Consolas", 11),
                                     fg="#00ffcc", bg="black", justify="center")
        self.result_label.place(relx=0.5, rely=0.85, anchor="center")

        # Footer
        self.footer_label = tk.Label(
            root,
            text="Team Members: Manasvi Badoni, Darshita Joshi,Uddeshya Jugran , Chetan Pandey",
            font=("Helvetica", 9),
            fg="#cccccc",
            bg="black"
        )
        self.footer_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

    def predict_churn(self):
        try:
            playtime = float(self.entries["PlayTimeHours"].get())
            purchases = int(self.entries["InGamePurchases"].get())
            level = int(self.entries["PlayerLevel"].get())
            achievements = int(self.entries["AchievementsUnlocked"].get())

            playtime_per_level = playtime / level if level > 0 else 0
            achievements_per_level = achievements / level if level > 0 else 0
            playtime_purchases_ratio = playtime / purchases if purchases > 0 else 0
            engagement_score = (
                playtime * 0.3 + purchases * 0.2 +
                achievements * 0.2 + (level / 10) * 0.3
            )
            engagement_level_num = 2 if engagement_score >= 20 else (
                1 if engagement_score >= 10 else 0)

            input_dict = {
                "PlayTimeHours": playtime,
                "InGamePurchases": purchases,
                "PlayerLevel": level,
                "AchievementsUnlocked": achievements,
                "PlaytimePerLevel": playtime_per_level,
                "AchievementsPerLevel": achievements_per_level,
                "PlaytimePurchasesRatio": playtime_purchases_ratio,
                "EngagementScore": engagement_score,
                "EngagementLevelNum": engagement_level_num
            }

            user_df = pd.DataFrame([input_dict])[features]
            prediction = clf.predict(user_df)[0]
            prob = clf.predict_proba(user_df)[0][1]
            label = ["Low", "Medium", "High"][engagement_level_num]
            churn_result = "Likely to Churn" if prediction == 1 else "Likely to Stay"

            self.result_label.config(
                text=f"Engagement Level: {label}\n"
                     f"Prediction: {churn_result}\n"
                     f"Churn Probability: {prob:.1%}"
            )

            visualize_engagement_playtime(input_dict, df_pd)

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input.\n{str(e)}")

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnGUI(root)
    root.mainloop()
    spark.stop()
