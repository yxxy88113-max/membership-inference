#!/usr/bin/env python3
"""PyTorch reproduction of the Shokri et al. membership inference pipeline."""
import argparse,json,random,time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

class MLP(nn.Module):
 def __init__(self,h=50): super().__init__(); self.net=nn.Sequential(nn.Linear(784,h),nn.Tanh(),nn.Linear(h,10))
 def forward(self,x): return self.net(x)

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def train(model,x,y,epochs,batch,lr,device):
 opt=torch.optim.Adam(model.parameters(),lr=lr); lossf=nn.CrossEntropyLoss(); hist=[]; ds=DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=True)
 for e in range(epochs):
  model.train(); total=correct=n=0
  for a,b in ds:
   a,b=a.to(device),b.to(device); opt.zero_grad(); z=model(a); loss=lossf(z,b); loss.backward(); opt.step(); total+=loss.item()*len(b); correct+=(z.argmax(1)==b).sum().item(); n+=len(b)
  hist.append((total/n,correct/n));
 return hist
@torch.no_grad()
def predict(model,x,device):
 model.eval(); return torch.softmax(model(x.to(device)),1).cpu().numpy()
def evalacc(p,y): return float(accuracy_score(y,p.argmax(1)))
def load_data(out, cfg, seedv, synthetic=False):
 if synthetic:
  rng=np.random.RandomState(seedv); n=cfg['target_train']+cfg['target_test']+cfg['shadow_train']+cfg['shadow_test']; x=rng.rand(n,784).astype('float32'); y=np.arange(n)%10; idx=rng.permutation(n); a,b,c,d=np.split(idx,[cfg['target_train'],cfg['target_train']+cfg['target_test'],cfg['target_train']+cfg['target_test']+cfg['shadow_train']]); np.savez(out/'split_indices.npz',target_train=a,target_test=b,shadow_pool_train=c,shadow_pool_test=d); return [(x[a],y[a]),(x[b],y[b]),(x[c],y[c]),(x[d],y[d])]
 from torchvision.datasets import MNIST
 from torchvision import transforms
 root=out/'data'; ds=MNIST(root=str(root),train=True,download=True,transform=transforms.ToTensor()); x=ds.data.numpy().astype('float32').reshape(-1,784)/255.; y=ds.targets.numpy().astype('int64')
 rng=np.random.RandomState(seedv); by=[np.where(y==c)[0] for c in range(10)]; rng.shuffle(by[0])
 # deterministic disjoint pools; balance labels as available
 idx=rng.permutation(len(y)); need=cfg['target_train']+cfg['target_test']+cfg['shadow_train']+cfg['shadow_test']; idx=idx[:need]
 a,b,c,d=np.split(idx,[cfg['target_train'],cfg['target_train']+cfg['target_test'],cfg['target_train']+cfg['target_test']+cfg['shadow_train']]);
 np.savez(out/'split_indices.npz',target_train=a,target_test=b,shadow_pool_train=c,shadow_pool_test=d)
 return [(x[a],y[a]),(x[b],y[b]),(x[c],y[c]),(x[d],y[d])]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.json'); ap.add_argument('--output',default='results'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--synthetic',action='store_true'); args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); cfg=json.load(open(args.config));
 if args.smoke: cfg.update(target_train=200,target_test=200,shadow_train=200,shadow_test=200,n_shadow=2,epochs=1)
 seed(cfg['seed']); device=torch.device('cpu'); t=time.time(); sets=load_data(out,cfg,cfg['seed'],args.synthetic); (tx,ty),(vx,vy),(sx,sy),(svx,svy)=sets
 target=MLP(cfg['hidden']).to(device); th=train(target,torch.tensor(tx),torch.tensor(ty),cfg['epochs'],cfg['batch_size'],cfg['lr'],device); tp=predict(target,torch.tensor(tx),device); vp=predict(target,torch.tensor(vx),device)
 torch.save(target.state_dict(),out/'target_model.pt'); rows=[]; rows += [{'epoch':i+1,'loss':l,'accuracy':a,'split':'target_train'} for i,(l,a) in enumerate(th)]; rows += [{'epoch':i+1,'loss':None,'accuracy':evalacc(tp,ty),'split':'target_train_eval'} for i in range(cfg['epochs'])]; rows += [{'epoch':i+1,'loss':None,'accuracy':evalacc(vp,vy),'split':'target_test_eval'} for i in range(cfg['epochs'])]; pd.DataFrame(rows).to_csv(out/'training_metrics.csv',index=False)
 fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(range(1,len(th)+1),[z[0] for z in th]); ax[0].set(title='Target loss',xlabel='epoch'); ax[1].plot(range(1,len(th)+1),[z[1] for z in th],label='train'); ax[1].axhline(evalacc(vp,vy),ls='--',label='test'); ax[1].legend(); ax[1].set(title='Target accuracy',xlabel='epoch'); fig.tight_layout(); fig.savefig(out/'target_training_curves.png'); plt.close(fig)
 attack_by_n=[]
 for nshadow in ([2,10] if cfg['n_shadow']>=10 else [cfg['n_shadow']]):
  ax=[]; ay=[]
  for i in range(nshadow):
   rng=np.random.RandomState(cfg['seed']+i); ii=rng.permutation(len(sx)); mid=len(ii)//2; a,b=ii[:mid],ii[mid:]; sm=MLP(cfg['hidden']).to(device); train(sm,torch.tensor(sx[a]),torch.tensor(sy[a]),cfg['epochs'],cfg['batch_size'],cfg['lr'],device); ax.extend([predict(sm,torch.tensor(sx[a]),device),predict(sm,torch.tensor(sx[b]),device)]); ay.extend([np.ones(len(a)),np.zeros(len(b))])
  attack_x=np.vstack(ax); attack_y=np.concatenate(ay).astype('int64'); txp=np.vstack([predict(target,torch.tensor(tx),device),predict(target,torch.tensor(vx),device)]); true=np.r_[np.ones(len(tx)),np.zeros(len(vx))].astype('int64'); pred=np.zeros(len(true),dtype='int64'); score=np.zeros(len(true),dtype='float32')
  for cls in range(10):
   tr=np.where(np.argmax(attack_x,1)==cls)[0]; te=np.where(np.argmax(txp,1)==cls)[0]
   
   if len(tr)<2 or len(te)==0: continue
   am=nn.Sequential(nn.Linear(10,50),nn.Tanh(),nn.Linear(50,2)).to(device); train(am,torch.tensor(attack_x[tr],dtype=torch.float32),torch.tensor(attack_y[tr]),cfg['epochs'],cfg['batch_size'],cfg['lr'],device); q=predict(am,torch.tensor(txp[te],dtype=torch.float32),device); pred[te]=q.argmax(1); score[te]=q[:,1]
  pred=np.array(pred); score=np.array(score); # class iteration order equals te order by construction
  m={'n_shadow':nshadow,'attack_accuracy':accuracy_score(true,pred),'precision':precision_recall_fscore_support(true,pred,average='binary',zero_division=0)[0],'recall':precision_recall_fscore_support(true,pred,average='binary',zero_division=0)[1],'f1':precision_recall_fscore_support(true,pred,average='binary',zero_division=0)[2],'roc_auc':roc_auc_score(true,score) if len(np.unique(true))>1 else .5}; attack_by_n.append(m)
  if nshadow==cfg['n_shadow']: pd.DataFrame({'true':true,'pred':pred,'score':score}).to_csv(out/'attack_predictions.csv',index=False); np.savez(out/'shadow_outputs.npz',probabilities=attack_x,labels=attack_y)
 pd.DataFrame(attack_by_n).to_csv(out/'attack_metrics.csv',index=False); fig,ax=plt.subplots(); ax.plot([0,1],[0,1],'--',label='random 50%');
 if len(score): fpr,tpr,_=roc_curve(true,score); ax.plot(fpr,tpr,label='attack'); ax.legend(); fig.savefig(out/'roc_curve.png'); plt.close(fig)
 cm=confusion_matrix(true,pred); fig,ax=plt.subplots(); ax.imshow(cm,cmap='Blues'); ax.set(xlabel='predicted',ylabel='true',title='Membership confusion matrix'); [ax.text(j,i,str(cm[i,j]),ha='center',va='center') for i in range(2) for j in range(2)]; fig.tight_layout(); fig.savefig(out/'attack_confusion_matrix.png'); plt.close(fig)
 print(json.dumps({'target_train_accuracy':evalacc(tp,ty),'target_test_accuracy':evalacc(vp,vy),'attack':attack_by_n,'seconds':time.time()-t},indent=2))
if __name__=='__main__': main()
