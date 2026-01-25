# test_gsheets.py
from gsheets import read_prompts_dataframe

if __name__ == "__main__":
    try:
        df = read_prompts_dataframe()
        total = len(df)
        print("✅ Successfully connected to Google Sheets via API")
        print(f"Loaded {total} row(s)\n")

        # Print sample in a nice table
        cols = ["prompt_id", "category", "question", "metric"]
        preview = df[cols].head(5).to_string(index=False, justify="left", max_colwidth=60)
        print("First 5 normalized rows:")
        print(preview)

        # Show unique metric types
        unique_metrics = df["metric"].apply(lambda m: m.split(":")[0] if ":" in m else m).unique()
        print("\nMetric types detected:", ", ".join(map(str, unique_metrics)))

        # Show category distribution
        top_cats = df["category"].value_counts().head(10)
        print("\nTop 10 categories:")
        for cat, count in top_cats.items():
            print(f"  {cat:<30} {count:>3}")

    except Exception as e:
        print("❌ Failed to load prompts from Google Sheets / CSV fallback")
        print("Error:", e)
