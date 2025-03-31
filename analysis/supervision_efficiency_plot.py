import pandas as pd
import matplotlib.pyplot as plt

for (layer, ecc) in [(3, 3), (2, 2), (3, 2)]:
    for nhead in [1, 16]:
        # Load the CSV
        df = pd.read_csv(f"/home/andy/Downloads/nhead_{nhead}_layer_{layer}_ecc_{ecc}.csv")

        # Plotting
        plt.figure(figsize=(5, 3))

        # Extract values
        x = df["Step"]

        # Without intermediate supervision
        y_false = df["model.intermediate_supervision: false - test_loss"]
        y_false_min = df["model.intermediate_supervision: false - test_loss__MIN"]
        y_false_max = df["model.intermediate_supervision: false - test_loss__MAX"]

        plt.plot(x, y_false, label="No Intermediate Supervision", color="blue")
        plt.fill_between(x, y_false_min, y_false_max, alpha=0.2, color="blue")

        # With intermediate supervision
        y_true = df["model.intermediate_supervision: true - test_loss"]
        y_true_min = df["model.intermediate_supervision: true - test_loss__MIN"]
        y_true_max = df["model.intermediate_supervision: true - test_loss__MAX"]

        plt.plot(x, y_true, label="With Intermediate Supervision", color="orange")
        plt.fill_between(x, y_true_min, y_true_max, alpha=0.2, color="orange")

        plt.ylim(0, 0.05)

        # Formatting
        plt.xlabel("Step")
        plt.ylabel("Test Loss")
        # plt.title("Test Loss Over Time with and without Intermediate Supervision")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

        plt.savefig(f"nhead_{nhead}_layer_{layer}_ecc_{ecc}.png", dpi=150)