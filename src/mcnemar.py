import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys

def load_preds(path):
    df = pd.read_csv(path)
    df['video_id'] = df['files'].apply(lambda x: '/'.join(str(x).replace('\\', '/').split('/')[:4]))
    df['prob'] = df['prob_class_1'].astype(float)
    df['label'] = df['labels'].astype(int)
    video_df = df.groupby('video_id').agg(
        true_label=('label', 'first'),
        mean_prob=('prob', 'mean')
    ).reset_index()
    video_df['pred'] = (video_df['mean_prob'] >= 0.5).astype(int)
    video_df['correct'] = (video_df['pred'] == video_df['true_label']).astype(int)
    return video_df

b16 = load_preds(sys.argv[1])
b32 = load_preds(sys.argv[2])
l14 = load_preds(sys.argv[3])

models = [("ViT-B/16", b16), ("ViT-B/32", b32), ("ViT-L/14", l14)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Video-Level Confusion Matrices — Celeb-DF-v2 Test Set", fontsize=14, fontweight='bold')

for ax, (name, df) in zip(axes, models):
    cm = confusion_matrix(df['true_label'], df['pred'])
    tn, fp, fn, tp = cm.ravel()
    ba = balanced_accuracy_score(df['true_label'], df['pred'])
    auc = roc_auc_score(df['true_label'], df['mean_prob'])
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{name}\nAUC={auc:.4f} | Bal.Acc={ba:.4f}", fontsize=11, fontweight='bold')
    
    classes = ['Real', 'Fake']
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")