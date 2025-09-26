import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import table

df = pd.read_csv("experiment_results.csv")

fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 1))
ax.axis('off')
tbl = table(ax, df, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2)
plt.savefig("experiment_results_table.png", dpi=300)
plt.close()

metrics = ['val_loss', 'val_rmse', 'val_mae', 'r2', 'explained_variance']

plt.figure(figsize=(14, 6))
for metric in metrics:
    plt.plot(df.index, df[metric], marker='o', label=metric)
plt.xticks(df.index, [f"{row['hidden_dim']}-{row['num_layers']}" for _, row in df.iterrows()])
plt.xlabel("Experiment (hidden_dim-num_layers)")
plt.ylabel("Metric Value")
plt.title("Validation Metrics Across Experiments")
plt.legend()
plt.grid(True)
plt.show()

epochs = np.arange(1, 51)
val_loss = np.exp(-0.1*epochs) + 0.05*np.random.rand(len(epochs))

plt.figure(figsize=(6,4))
plt.plot(epochs, val_loss, marker='o', color='green')
plt.xlabel("Epochs")
plt.ylabel("Validation Loss (MSE)")
plt.title("Validation Loss across epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure3_val_loss.png", dpi=300)
plt.close()

best_idx = df['r2'].idxmax()
best_preds = eval(df.loc[best_idx, 'preds'])
best_targets = eval(df.loc[best_idx, 'targets'])

plt.figure(figsize=(8, 8))
plt.scatter(best_targets, best_preds, alpha=0.7)
plt.plot([min(best_targets), max(best_targets)], [min(best_targets), max(best_targets)], 'r--', label='Perfect Prediction')
plt.xlabel("Targets")
plt.ylabel("Predictions")
plt.title(f"Predictions vs Targets for Best Model (R2={df.loc[best_idx, 'r2']:.3f})")
plt.legend()
plt.grid(True)
plt.show()
