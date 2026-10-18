#!/usr/bin/env python3
"""Download and validate the four MNIST IDX gzip files from a public mirror."""
import argparse, gzip, hashlib, struct
from pathlib import Path
from urllib.request import urlopen
FILES={
 'train-images-idx3-ubyte.gz':(2051,60000,28,28),
 'train-labels-idx1-ubyte.gz':(2049,60000),
 't10k-images-idx3-ubyte.gz':(2051,10000,28,28),
 't10k-labels-idx1-ubyte.gz':(2049,10000),
}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',default='results_mnist_10/data/MNIST/raw'); a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 for name,expected in FILES.items():
  dst=out/name; url='https://raw.githubusercontent.com/fgnt/mnist/master/'+name
  print('Downloading',name)
  with urlopen(url,timeout=60) as r, open(dst,'wb') as f:
   while True:
    b=r.read(1024*1024)
    if not b: break
    f.write(b)
  with gzip.open(dst,'rb') as f: header=f.read(16)
  magic,count=struct.unpack('>II',header[:8]); actual=(magic,count)
  if magic==2051: actual+=(struct.unpack('>II',header[8:16]))
  if actual != expected: raise RuntimeError('%s header %s != expected %s' % (name,actual,expected))
  digest=hashlib.sha256(dst.read_bytes()).hexdigest()
  print('  OK:',actual,'SHA256:',digest)
 print('MNIST download and validation complete:',out)
if __name__=='__main__': main()
