#!/usr/bin/env python3
"""Strict MNIST membership inference: official test non-members and true-label attacks."""
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score
from torchvision.datasets import MNIST
from torchvision import transforms
from run_experiment import MLP,train,predict

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def load_split(root,cfg,seed):
 tr=MNIST(str(root),train=True,download=False,transform=transforms.ToTensor()); te=MNIST(str(root),train=False,download=False,transform=transforms.ToTensor())
 x=tr.data.numpy().astype('float32').reshape(-1,784)/255.; y=tr.targets.numpy().astype('int64'); tx=te.data.numpy().astype('float32').reshape(-1,784)/255.; ty=te.targets.numpy().astype('int64')
 rng=np.random.RandomState(seed); ti=rng.permutation(len(y)); vi=rng.permutation(len(ty)); n=cfg['target_train']; m=cfg['target_test']; a=ti[:n]; pool=ti[n:]; b=vi[:m]; need=cfg['shadow_train']+cfg['shadow_test']; pool=pool[:need]; c,d=np.split(pool,[cfg['shadow_train']])
 return (x[a],y[a]),(tx[b],ty[b]),(x[c],y[c]),(x[d],y[d]),{'target_train':a,'target_test':b,'shadow_train':c,'shadow_test':d,'target_train_source':'MNIST train','target_test_source':'MNIST test','shadow_train_source':'MNIST train remainder','shadow_test_source':'MNIST train remainder'}

def metrics(y,p,s):
 pr,re,f,_=precision_recall_fscore_support(y,p,labels=[1],average=None,zero_division=0); pm,rm,fm,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
 return {'attack_accuracy':float(accuracy_score(y,p)),'member_precision':float(pr[0]),'member_recall':float(re[0]),'member_f1':float(f[0]),'macro_precision':float(pm),'macro_recall':float(rm),'macro_f1':float(fm),'roc_auc':float(roc_auc_score(y,s))}
def threshold_attack(shadow_score,shadow_y,target_score):
 vals=np.unique(shadow_score); candidates=np.r_[vals, (vals[:-1]+vals[1:])/2] if len(vals)>1 else vals; best=max(candidates,key=lambda t:accuracy_score(shadow_y,(shadow_score>=t).astype('int64'))); p=(target_score>=best).astype('int64'); return p,target_score, float(best)
def class_attack(sx,sy,svx,svy,target_prob,target_lab,nshadow,cfg,device,seed,truth):
 ax=[]; ay=[]; al=[]
 for i in range(nshadow):
  a=np.random.RandomState(seed+10000+i).permutation(len(sx)); b=np.random.RandomState(seed+20000+i).permutation(len(svx)); sm=MLP(cfg['hidden']).to(device); seed_all(seed+10000+i); train(sm,torch.tensor(sx[a]),torch.tensor(sy[a]),cfg['epochs'],cfg['batch_size'],cfg['lr'],device)
  ax.extend([predict(sm,torch.tensor(sx[a]),device),predict(sm,torch.tensor(svx[b]),device)]); ay.extend([np.ones(len(a),dtype='int64'),np.zeros(len(b),dtype='int64')]); al.extend([sy[a],svy[b]])
 attack_x=np.vstack(ax); attack_y=np.concatenate(ay); attack_lab=np.concatenate(al); pred=np.zeros(len(target_lab),dtype='int64'); score=np.zeros(len(target_lab),dtype='float32')
 per=[]
 for cls in range(10):
  tr=np.where(attack_lab==cls)[0]; te=np.where(target_lab==cls)[0]
  if len(tr)<2 or len(te)==0: continue
  am=nn.Sequential(nn.Linear(10,50),nn.Tanh(),nn.Linear(50,2)).to(device); seed_all(seed+20000+cls); train(am,torch.tensor(attack_x[tr],dtype=torch.float32),torch.tensor(attack_y[tr]),cfg.get('attack_epochs',cfg['epochs']),cfg['batch_size'],cfg['lr'],device); q=predict(am,torch.tensor(target_prob[te],dtype=torch.float32),device); pred[te]=q.argmax(1); score[te]=q[:,1]; per.append({'class':cls,'n_target':len(te),**metrics(truth[te],pred[te],score[te])})
 return metrics(truth,pred,score),pred,score,attack_x,attack_y,attack_lab,per

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.json'); ap.add_argument('--data-root',default='results_mnist_10/data'); ap.add_argument('--output',default='rigorous_results_v2'); ap.add_argument('--epochs',default='20,50'); ap.add_argument('--seeds',default='21312,21313,21314,21315,21316'); args=ap.parse_args(); cfg=json.load(open(args.config)); epochs=[int(x) for x in args.epochs.split(',')]; seeds=[int(x) for x in args.seeds.split(',')]; out=Path(args.output); out.mkdir(parents=True,exist_ok=True); device=torch.device('cpu'); rows=[]; curves=[]; start=time.time()
 per_rows=[]
 for ep in epochs:
  c=dict(cfg,epochs=ep,attack_epochs=min(5,ep))
  for s in seeds:
   seed_all(s); (tx,ty),(vx,vy),(sx,sy),(svx,svy),idx=load_split(args.data_root,c,s); target=MLP(c['hidden']).to(device); hist=train(target,torch.tensor(tx),torch.tensor(ty),ep,c['batch_size'],c['lr'],device); a=predict(target,torch.tensor(tx),device); b=predict(target,torch.tensor(vx),device); ta=float(accuracy_score(ty,a.argmax(1))); va=float(accuracy_score(vy,b.argmax(1))); curves += [{'seed':s,'epochs':ep,'epoch':i+1,'loss':h[0],'train_accuracy':h[1],'test_accuracy':va} for i,h in enumerate(hist)]; target_prob=np.vstack([a,b]); target_lab=np.r_[ty,vy]; truth=np.r_[np.ones(len(ty),dtype='int64'),np.zeros(len(vy),dtype='int64')]
   for nshadow in (2,10):
    # true-label per-class attack
    m,pred,score,ax,ay,al,per=class_attack(sx,sy,svx,svy,target_prob,target_lab,nshadow,c,device,s,truth)
    per_rows += [dict(seed=s,epochs=ep,n_shadow=nshadow,attack_type='true_label_neural',**q) for q in per]
    m.update(seed=s,epochs=ep,n_shadow=nshadow,target_train_accuracy=ta,target_test_accuracy=va,generalization_gap=ta-va,attack_type='true_label_neural'); rows.append(m)
    max_shadow=ax.max(1); max_target=target_prob.max(1); p0,sc,t=threshold_attack(max_shadow,ay,max_target); z=metrics(truth,p0,sc); z.update(seed=s,epochs=ep,n_shadow=nshadow,target_train_accuracy=ta,target_test_accuracy=va,generalization_gap=ta-va,attack_type='max_confidence'); rows.append(z)
    ent_shadow=-(ax*np.log(np.clip(ax,1e-12,1))).sum(1); ent_target=-(target_prob*np.log(np.clip(target_prob,1e-12,1))).sum(1); p0,sc,t=threshold_attack(-ent_shadow,ay,-ent_target); z=metrics(truth,p0,sc); z.update(seed=s,epochs=ep,n_shadow=nshadow,target_train_accuracy=ta,target_test_accuracy=va,generalization_gap=ta-va,attack_type='entropy'); rows.append(z)
    loss_shadow=-np.log(np.clip(ax[np.arange(len(ax)),al],1e-12,1)); loss_target=-np.log(np.clip(target_prob[np.arange(len(target_prob)),target_lab],1e-12,1)); p0,sc,t=threshold_attack(-loss_shadow,ay,-loss_target); z=metrics(truth,p0,sc); z.update(seed=s,epochs=ep,n_shadow=nshadow,target_train_accuracy=ta,target_test_accuracy=va,generalization_gap=ta-va,attack_type='cross_entropy'); rows.append(z)
   torch.save(target.state_dict(),out/f'target_seed{s}_ep{ep}.pt'); np.savez(out/f'split_seed{s}.npz',target_train=idx['target_train'],target_test=idx['target_test'],shadow_train=idx['shadow_train'],shadow_test=idx['shadow_test'])
 df=pd.DataFrame(rows); df.to_csv(out/'results.csv',index=False); pd.DataFrame(per_rows).to_csv(out/'per_class_results.csv',index=False); df.groupby(['epochs','n_shadow','attack_type']).agg(['mean','std']).to_csv(out/'summary.csv'); pd.DataFrame(curves).to_csv(out/'target_curves.csv',index=False); json.dump({'target_test_source':'MNIST test','shadow_source':'MNIST train remainder','seeds':seeds},open(out/'data_audit.json','w'),indent=2); print(df.groupby(['epochs','n_shadow','attack_type'])[['attack_accuracy','member_f1','macro_f1','roc_auc']].mean()); print('seconds',time.time()-start)
if __name__=='__main__': main()
