import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def predict_churn_from_input(clf, features, df_pd):
    print("\n--- Enter Gameplay Details ---")
    try:
        playtime_hours = float(input("Playtime in last 30 days (hours): "))
        in_game_purchases = int(input("Number of game purchases: "))
        level = int(input("Player Level: "))
        is_male = input("Gender : Male or Female : ")
        achievements = int(input("Number of Achievements Unlocked: "))
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return

    playtime_per_level = playtime_hours / level if level > 0 else 0
    achievements_per_level = achievements / level if level > 0 else 0
    playtime_purchases_ratio = playtime_hours / in_game_purchases if in_game_purchases > 0 else 0

    engagement_score = (
        playtime_hours * 0.3 +
        in_game_purchases * 0.2 +
        achievements * 0.2 +
        (level / 10) * 0.3
    )

    engagement_level_num = 2 if engagement_score >= 20 else 1 if engagement_score >= 10 else 0

    user_input = {
        "PlayTimeHours": playtime_hours,
        "InGamePurchases": in_game_purchases,
        "PlayerLevel": level,
        "AchievementsUnlocked": achievements,
        "PlaytimePerLevel": playtime_per_level,
        "AchievementsPerLevel": achievements_per_level,
        "PlaytimePurchasesRatio": playtime_purchases_ratio,
        "EngagementScore": engagement_score,
        "EngagementLevelNum": engagement_level_num
    }

    user_df = pd.DataFrame([user_input])[features]
    prediction = clf.predict(user_df)[0]
    engagement_label = {0: "Low", 1: "Medium", 2: "High"}[engagement_level_num]
    
    print(f"\n Engagement Level: {engagement_label}")
    print("\n Prediction:", "Likely to churn." if prediction == 1 else "Likely to stay engaged!")

    visualize_engagement_playtime(user_input, df_pd)

def visualize_engagement_playtime(user_input_row, df_pd):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df_pd["EngagementScore"], bins=30, kde=True, ax=axs[0], color='skyblue')
    axs[0].axvline(user_input_row["EngagementScore"], color='red', linestyle='--', label="User Input")
    axs[0].set_title("Engagement Score Distribution")

    sns.histplot(df_pd["PlayTimeHours"], bins=30, kde=True, ax=axs[1], color='lightgreen')
    axs[1].axvline(user_input_row["PlayTimeHours"], color='red', linestyle='--', label="User Input")
    axs[1].set_title("PlayTime Hours Distribution")

    for ax in axs:
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.suptitle("User Gameplay Stats vs Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
