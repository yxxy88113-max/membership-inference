#!/usr/bin/env python3
"""Controlled MNIST experiment: same target per seed/epoch, varying only shadow count."""
import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torchvision.datasets import MNIST
from torchvision import transforms
from run_experiment import MLP, train, predict

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def load_split(root, cfg, seed):
    ds=MNIST(root=str(root), train=True, download=False, transform=transforms.ToTensor())
    x=ds.data.numpy().astype('float32').reshape(-1,784)/255.; y=ds.targets.numpy().astype('int64')
    rng=np.random.RandomState(seed); idx=rng.permutation(len(y)); n=cfg['target_train']+cfg['target_test']+cfg['shadow_train']+cfg['shadow_test']; idx=idx[:n]
    a,b,c,d=np.split(idx,[cfg['target_train'],cfg['target_train']+cfg['target_test'],cfg['target_train']+cfg['target_test']+cfg['shadow_train']])
    return (x[a],y[a]),(x[b],y[b]),(x[c],y[c]),(x[d],y[d]), {'target_train':a,'target_test':b,'shadow_train':c,'shadow_test':d}

def class_attack(shadow_x, shadow_y, target_prob, target_truth, nshadow, cfg, device, seed):
    ax=[]; ay=[]
    for i in range(nshadow):
        rng=np.random.RandomState(seed+10000+i); ii=rng.permutation(len(shadow_x)); mid=len(ii)//2; a,b=ii[:mid],ii[mid:]
        sm=MLP(cfg['hidden']).to(device)
        seed_all(seed+10000+i)
        train(sm,torch.tensor(shadow_x[a]),torch.tensor(shadow_y[a]),cfg['epochs'],cfg['batch_size'],cfg['lr'],device)
        ax.extend([predict(sm,torch.tensor(shadow_x[a]),device),predict(sm,torch.tensor(shadow_x[b]),device)])
        ay.extend([np.ones(len(a),dtype='int64'),np.zeros(len(b),dtype='int64')])
    attack_x=np.vstack(ax); attack_y=np.concatenate(ay)
    pred=np.zeros(len(target_truth),dtype='int64'); score=np.zeros(len(target_truth),dtype='float32')
    for cls in range(10):
        tr=np.where(np.argmax(attack_x,1)==cls)[0]; te=np.where(np.argmax(target_prob,1)==cls)[0]
        if len(tr)<2 or len(te)==0: continue
        am=nn.Sequential(nn.Linear(10,50),nn.Tanh(),nn.Linear(50,2)).to(device)
        seed_all(seed+20000+cls)
        train(am,torch.tensor(attack_x[tr],dtype=torch.float32),torch.tensor(attack_y[tr]),cfg['epochs'],cfg['batch_size'],cfg['lr'],device)
        q=predict(am,torch.tensor(target_prob[te],dtype=torch.float32),device); pred[te]=q.argmax(1); score[te]=q[:,1]
    p,r,f,_=precision_recall_fscore_support(target_truth,pred,labels=[1],average=None,zero_division=0)
    pm,rm,fm,_=precision_recall_fscore_support(target_truth,pred,average='macro',zero_division=0)
    return {'attack_accuracy':float(accuracy_score(target_truth,pred)), 'member_precision':float(p[0]),'member_recall':float(r[0]),'member_f1':float(f[0]),'macro_precision':float(pm),'macro_recall':float(rm),'macro_f1':float(fm),'roc_auc':float(roc_auc_score(target_truth,score))}, pred, score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.json'); ap.add_argument('--data-root',default='results_mnist_10/data'); ap.add_argument('--output',default='rigorous_results'); ap.add_argument('--epochs',default='20,50'); ap.add_argument('--seeds',default='21312,21313,21314'); args=ap.parse_args()
    cfg=json.load(open(args.config)); epochs_list=[int(x) for x in args.epochs.split(',')]; seeds=[int(x) for x in args.seeds.split(',')]; out=Path(args.output); out.mkdir(parents=True,exist_ok=True); device=torch.device('cpu'); allrows=[]; curves=[]; started=time.time()
    for ep in epochs_list:
      cfg_ep=dict(cfg); cfg_ep['epochs']=ep
      for s in seeds:
        seed_all(s); (tx,ty),(vx,vy),(sx,sy),(svx,svy),indices=load_split(args.data_root,cfg_ep,s)
        target=MLP(cfg['hidden']).to(device); hist=train(target,torch.tensor(tx),torch.tensor(ty),ep,cfg['batch_size'],cfg['lr'],device)
        ttrain=predict(target,torch.tensor(tx),device); ttest=predict(target,torch.tensor(vx),device); train_acc=float(accuracy_score(ty,ttrain.argmax(1))); test_acc=float(accuracy_score(vy,ttest.argmax(1)))
        curves += [{'seed':s,'epochs':ep,'epoch':i+1,'loss':z[0],'train_accuracy':z[1],'test_accuracy':test_acc} for i,z in enumerate(hist)]
        target_prob=np.vstack([ttrain,ttest]); truth=np.r_[np.ones(len(tx),dtype='int64'),np.zeros(len(vx),dtype='int64')]
        for nshadow in (2,10):
          m,pred,score=class_attack(sx,sy,target_prob,truth,nshadow,cfg_ep,device,s)
          allrows.append({'seed':s,'epochs':ep,'n_shadow':nshadow,'target_train_accuracy':train_acc,'target_test_accuracy':test_acc,'generalization_gap':train_acc-test_acc,**m})
        torch.save(target.state_dict(),out/f'target_seed{s}_ep{ep}.pt')
        np.savez(out/f'split_seed{s}.npz',**indices)
    df=pd.DataFrame(allrows); df.to_csv(out/'controlled_results.csv',index=False); summary=df.groupby(['epochs','n_shadow']).agg(['mean','std']); summary.to_csv(out/'controlled_summary.csv'); pd.DataFrame(curves).to_csv(out/'target_curves.csv',index=False)
    print(df.to_string(index=False)); print('\nSUMMARY\n',summary.to_string()); print('seconds',time.time()-started)
if __name__=='__main__': main()
