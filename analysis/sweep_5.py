import pandas as pd

# Load data from CSV
df = pd.read_csv("../results/sweep_5_node_result.csv")

# Sort by 'model.num_layers' and 'model.nhead'
df_sorted = df.sort_values(by=["model.num_layers", "model.nhead"])

# De-duplicate rows based on both columns, keeping the first occurrence
df_dedup = df_sorted.drop_duplicates(subset=["model.num_layers", "model.nhead"])

# Optionally, reset the index
df_dedup = df_dedup.reset_index(drop=True)
with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    1000,
    "display.precision",
    2,
    "display.colheader_justify",
    "center",
):
    print(df_dedup)

# Select rows where 'test_loss' < 0.05
df_filtered = df_dedup[df_dedup["test_loss"] < 0.05]

print(df_filtered)
#
#
#
# # save back to CSV
# df_dedup.to_csv('../results/sweep_5_node_result_dedup.csv', index=False)
