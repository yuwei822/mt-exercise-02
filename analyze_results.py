import matplotlib.pyplot as plt
import pandas as pd
import re
import os

# different dropout
dropouts = [0, 0.2, 0.4, 0.6, 0.8]

# extract data of perplexity
def extract_ppl(logfile):
    epochs, valid_ppl = [], []
    with open(logfile, 'r') as file:
        for line in file:
            match = re.search(r'end of epoch\s+(\d+).*valid ppl\s+([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                valid_ppl.append(float(match.group(2)))
    return epochs, valid_ppl

# create dataframe
data = {}
for d in dropouts:
    filename = f'logs/dropout_{str(d).replace(".", "_")}.log'
    epochs, valid_ppl = extract_ppl(filename)
    data[f'Dropout {d}'] = valid_ppl

df = pd.DataFrame(data, index=epochs)

# save the data to csv
df.to_csv('perplexity_summary.csv')
print("Table saved to perplexity_summary.csv")

# make the plot
plt.figure(figsize=(10,6))
for col in df.columns:
    plt.plot(df.index, df[col], marker='o', label=col)

plt.xlabel('Epochs')
plt.ylabel('Validation Perplexity')
plt.title('Validation Perplexity vs. Dropout Rate')
plt.legend()
plt.grid(True)
plt.savefig('validation_perplexity_plot.png')
print("Plot saved to validation_perplexity_plot.png")
