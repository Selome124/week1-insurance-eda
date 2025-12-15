import pandas as pd

# Load scraped reviews
df = pd.read_csv("reviews.csv")

# ----------------------------
# 1. REMOVE DUPLICATES
# ----------------------------
df.drop_duplicates(subset=["reviewId", "content"], inplace=True)

# ----------------------------
# 2. HANDLE MISSING DATA
# ----------------------------

# Drop rows with no review text or no rating
df.dropna(subset=["content", "score"], inplace=True)

# Fill missing dates with a placeholder (unlikely, but just in case)
df["at"] = df["at"].fillna("2000-01-01")

# ----------------------------
# 3. NORMALIZE DATES
# ----------------------------
df["date"] = pd.to_datetime(df["at"]).dt.strftime("%Y-%m-%d")

# ----------------------------
# 4. RENAME / SELECT COLUMNS
# ----------------------------
df["review"] = df["content"]
df["rating"] = df["score"]
df["bank"] = df["app_name"]
df["source"] = "Google Play"

clean_df = df[["review", "rating", "date", "bank", "source"]]

# ----------------------------
# 5. SAVE CLEAN CSV
# ----------------------------
clean_df.to_csv("clean_reviews.csv", index=False)

print("✨ Preprocessing complete! Saved to clean_reviews.csv")
