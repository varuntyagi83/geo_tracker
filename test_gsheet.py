# test_gsheets.py
import sys
from textwrap import shorten
from gsheets import read_prompts_dataframe

def main():
    try:
        df = read_prompts_dataframe()
        total = len(df)
        colmap = df.attrs.get("colmap", {})

        print("✅ Successfully connected and normalized Google Sheet")
        print(f"Loaded {total} row(s)\n")

        # Show original columns and the chosen mapping
        print("Original sheet columns:")
        print("  ", ", ".join(map(str, colmap.get("all_columns", []))) or "(unknown)")
        print("\nChosen mapping (sheet → canonical):")
        print(f"  question  ← {colmap.get('question_col')}")
        print(f"  category  ← {colmap.get('category_col')}")
        print(f"  prompt_id ← {colmap.get('prompt_id_col')}")
        print(f"  metric    ← {colmap.get('metric_col')}  (always ignored / empty)")
        print(f"  keywords  ← {colmap.get('keywords_col')} (tracked only, not returned)")
        print()

        # Verify metric column is empty (as intended)
        non_empty_metric = df["metric"].astype(str).str.strip().ne("")
        n_non_empty = int(non_empty_metric.sum())
        if n_non_empty == 0:
            print("ℹ️  No per-row metric present (as intended). Presence/Sentiment are skipped for all rows.")
        else:
            print(f"⚠️  Found {n_non_empty} row(s) with a non-empty 'metric' value. This would trigger presence/sentiment.")

        print("\nFirst 5 normalized rows (what will be forwarded to the LLM):")
        for _, r in df.head(5).iterrows():
            pid = r["prompt_id"]
            cat = r["category"]
            q   = r["question"]
            print(f"  {pid:<6} | {cat:<20} | Q: {shorten(q, 160, placeholder='…')}")

        print("\nEverything looks good ✅")

    except Exception as e:
        print("❌ Failed to load/normalize prompts from Google Sheets / CSV fallback")
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
