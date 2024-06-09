import torch
from bert_score import score, plot_example
import matplotlib.pyplot as plt

# data
cands = ['Several cell types in a tissue proliferation IL-8 .']
refs = ['tumor cell density is a basic pathological feature of solid tumors.']

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# calculate BERTScore
P, R, F1 = score(cands, refs, lang="en", verbose=True, device=device)
print(f"System level P: {P.mean():.4f}")
print(f"System level R: {R.mean():.4f}")
print(f"System level F1 score: {F1.mean():.4f}")

# Set all font sizes to 14
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['font.family'] = 'SimHei'  # Also setting the font family here

# plot example
cand = cands[0]
ref = refs[0]
plot_example(cand, ref, lang="en")
