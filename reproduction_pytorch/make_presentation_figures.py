#!/usr/bin/env python3
"""Create thesis/presentation figures from already executed experiment CSV files."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
ROOT=Path(__file__).parent; V2=ROOT/'rigorous_results_v2'; OUT=ROOT/'presentation_figures'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(V2/'results.csv'); cur=pd.read_csv(V2/'target_curves.csv')
# 1: controlled target learning curves
fig,ax=plt.subplots(1,2,figsize=(11,4))
for seed,g in cur.groupby('seed'):
 ax[0].plot(g.epoch,g.loss,alpha=.65,label=str(seed)); ax[1].plot(g.epoch,g.train_accuracy,alpha=.65,label=str(seed))
ax[0].set(xlabel='Epoch',ylabel='Training loss',title='Target model training loss')
ax[1].set(xlabel='Epoch',ylabel='Accuracy',title='Target model training accuracy'); ax[1].axhline(df.target_test_accuracy.mean(),ls='--',color='black',label='test mean')
ax[0].legend(title='seed'); ax[1].legend(); fig.tight_layout(); fig.savefig(OUT/'fig1_target_training.png'); plt.close(fig)
# 2: attack method comparison
agg=df.groupby(['n_shadow','attack_type']).agg(accuracy=('attack_accuracy','mean'),auc=('roc_auc','mean'),f1=('member_f1','mean')).reset_index(); methods=list(agg.attack_type.unique()); names={'true_label_neural':'Neural (true label)','max_confidence':'Max confidence','entropy':'Entropy','cross_entropy':'Cross-entropy'}
fig,ax=plt.subplots(1,3,figsize=(14,4)); x=np.arange(len(methods)); w=.35
for j,m in enumerate(['accuracy','f1','auc']):
 for k,n in enumerate([2,10]):
  z=agg[agg.n_shadow==n].set_index('attack_type').reindex(methods)[m]; ax[j].bar(x+(k-.5)*w,z,width=w,label=f'{n} shadows')
 ax[j].axhline(.5,ls='--',color='gray'); ax[j].set_xticks(x,names.values(),rotation=25,ha='right'); ax[j].set_ylim(0,1); ax[j].set_title({'accuracy':'Attack accuracy','f1':'Member F1','auc':'ROC-AUC'}[m]); ax[j].set_ylabel('score')
ax[0].legend(); fig.tight_layout(); fig.savefig(OUT/'fig2_attack_methods.png'); plt.close(fig)
# 3: target generalization gap
z=df[df.attack_type=='true_label_neural'].groupby('n_shadow')[['target_train_accuracy','target_test_accuracy','generalization_gap']].mean().reset_index(); fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(z)); ax.bar(x-.25,z.target_train_accuracy,width=.25,label='train'); ax.bar(x,z.target_test_accuracy,width=.25,label='test'); ax.bar(x+.25,z.generalization_gap,width=.25,label='gap'); ax.set_xticks(x,['2 shadows','10 shadows']); ax.set_ylim(0,1.05); ax.set_title('Target performance and generalization gap'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'fig3_generalization.png'); plt.close(fig)
# 4: methodology diagram
fig,ax=plt.subplots(figsize=(12,3)); ax.axis('off'); boxes=[('MNIST train','target members\nshadow data'),('Target model','784 → 50 tanh → 10'),('Probability','10-class softmax'),('Attack models','one model per\ntrue digit class'),('Metrics','Accuracy / F1 / ROC-AUC')]; xs=np.linspace(.08,.92,len(boxes))
for i,(x,(title,body)) in enumerate(zip(xs,boxes)):
 ax.text(x,.52,title,ha='center',va='center',fontsize=13,weight='bold',bbox=dict(boxstyle='round,pad=.7',facecolor='#d9eaf7',edgecolor='#2b6c99')); ax.text(x,.18,body,ha='center',va='center',fontsize=10)
 if i<len(boxes)-1: ax.annotate('',xy=(xs[i+1]-.08,.52),xytext=(x+.08,.52),arrowprops=dict(arrowstyle='->',lw=2))
ax.set_title('Membership inference reproduction pipeline',fontsize=15); fig.tight_layout(); fig.savefig(OUT/'fig4_pipeline.png'); plt.close(fig)
print('generated',*[p.name for p in OUT.glob('*.png')])
