import matplotlib.pyplot as plt

# ROC points: (FPR, TPR)
fpr = [0.0, 0.0, 0.011, 0.011, 0.011, 0.022, 0.044, 1.0]
tpr = [0.0, 0.683, 0.7, 0.7, 0.7, 0.717, 0.767, 1.0]

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, marker='o', label='ROC Curve (Manual)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Task 2a (Manual Calculation)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()