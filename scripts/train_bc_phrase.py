#!/usr/bin/env python3
"""Train a causal BC-style keyword detector on the prepared phrase corpus."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SR=16000; CLIP=9600

def read(p):
    x,r=sf.read(p,dtype='float32'); assert r==SR; return torch.as_tensor(x).flatten()
def segments(x):
    e=x.unfold(0,320,160).square().mean(-1).sqrt(); a=e>.006; out=[]; s=None
    for i,v in enumerate(a.tolist()+[False]):
        if v and s is None:s=i
        if not v and s is not None:
            l=max(0,s*160-960); q=min(x.numel(),i*160+960)
            if .18<=(q-l)/SR<=1.5:out.append((l,q))
            s=None
    return out
def crop(x,left):
    y=torch.zeros(CLIP); l=max(0,left); r=min(x.numel(),left+CLIP)
    if r>l:y[l-left:r-left]=x[l:r]
    return y
class DS(torch.utils.data.Dataset):
    def __init__(self,p,sp,n,count,seed,aug):self.p=p;self.sp=sp;self.n=n;self.count=count;self.seed=seed;self.aug=aug
    def __len__(self):return self.count
    def __getitem__(self,i):
        g=random.Random(self.seed+i*10007); pos=i%2==0
        if pos:
            l,r=g.choice(self.sp); x=crop(self.p,(l+r)//2-CLIP//2+g.randint(-1500,1500)); y=1
        else:x=crop(self.n,g.randrange(max(1,self.n.numel()-CLIP)));y=0
        if self.aug:
            x=torch.roll(x,g.randint(-700,700))*g.uniform(.7,1.3)
            x=(x+torch.randn_like(x)*g.uniform(.0002,.003)).clamp(-1,1)
        return x,y
class Block(nn.Module):
    def __init__(self,c,d):
        super().__init__(); self.d=d; self.dw=nn.Conv1d(c,c,5,groups=c,dilation=d,bias=False);self.bn=nn.BatchNorm1d(c);self.pw=nn.Conv1d(c,c,1,bias=False);self.out=nn.BatchNorm1d(c)
    def forward(self,x):
        z=F.pad(x,((5-1)*self.d,0));z=F.silu(self.bn(self.dw(z)));z=self.out(self.pw(z));return F.silu(x+z)
class BCResPhraseNet(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('window',torch.hann_window(400),persistent=False)
        self.inp=nn.Sequential(nn.Conv1d(40,24,3),nn.BatchNorm1d(24),nn.SiLU())
        self.blocks=nn.Sequential(Block(24,1),Block(24,2),Block(24,4),Block(24,8),Block(24,16))
        self.head=nn.Sequential(nn.Conv1d(24,48,1),nn.SiLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(48,2))
    def forward(self,x):
        s=torch.stft(x,n_fft=400,hop_length=160,win_length=400,window=self.window.to(x.device),return_complex=True).abs().square().clamp_min(1e-7).log()
        s=F.adaptive_avg_pool2d(s.unsqueeze(1),(40,s.shape[-1])).squeeze(1)
        return self.head(self.blocks(self.inp(s)))
@torch.no_grad()
def evalm(m,ds,dev):
    p=[];y=[]
    for x,l in torch.utils.data.DataLoader(ds,batch_size=128):p += m(x.to(dev)).softmax(1)[:,1].cpu().tolist();y += l.tolist()
    p=torch.tensor(p);y=torch.tensor(y);best=None
    for t in torch.linspace(.5,.999,100):
        q=p>=t;tp=((q)&(y==1)).sum().item();fp=((q)&(y==0)).sum().item();fn=((~q)&(y==1)).sum().item();rec=tp/max(1,tp+fn);pre=tp/max(1,tp+fp);f=2*rec*pre/max(1e-9,rec+pre);score=f-.25*fp/max(1,(y==0).sum().item())
        if best is None or score>best[0]:best=(score,float(t),{'threshold':float(t),'recall':rec,'precision':pre,'f1':f,'false_positives':fp})
    return best[1],best[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--positive-train',required=True);p.add_argument('--positive-validation',required=True);p.add_argument('--negative-train',required=True);p.add_argument('--negative-validation',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=25);a=p.parse_args();random.seed(7);torch.manual_seed(7);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt=read(a.positive_train);pv=read(a.positive_validation);nt=read(a.negative_train);nv=read(a.negative_validation);sp=segments(pt);sv=segments(pv);tr=DS(pt,sp,nt,4096,7,True);va=DS(pv,sv,nv,2048,8,False);m=BCResPhraseNet().to(dev);o=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);best=(-1,None,None)
    for ep in range(a.epochs):
        m.train();ls=[]
        for x,y in torch.utils.data.DataLoader(tr,batch_size=64,shuffle=True):
            z=F.cross_entropy(m(x.to(dev)),y.to(dev));o.zero_grad();z.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);o.step();ls.append(z.item())
        m.eval();t,met=evalm(m,va,dev);sc=met['f1']-.25*met['false_positives']/1024
        if sc>best[0]:best=(sc,t,{k:v for k,v in met.items()});state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        print(json.dumps({'epoch':ep+1,'loss':sum(ls)/len(ls),**met}),flush=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'format':'hushmic-bc-phrase-net-v1','sample_rate':SR,'clip_samples':CLIP,'phrase':'иди нахуй','threshold':best[1],'model':state,'validation':best[2]},out);print(json.dumps({'saved':str(out),'best':best[2]}))
if __name__=='__main__':main()
