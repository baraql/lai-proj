import re
import numpy as np
import matplotlib.pyplot as plt

loss_values = []
with open('logs/lsai_baseline-454785.out', 'r') as f:
    for line in f:
        match = re.search(r'Loss: ([0-9]+\.[0-9]+)', line)
        if match:
            loss_values.append(float(match.group(1)))

loss_values = loss_values[::2] 
steps = np.arange(5, 5 * len(loss_values) + 1, 5)

plt.figure(figsize=(10, 6))
plt.plot(steps, loss_values, label='Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png', bbox_inches='tight')
plt.close()
