import sys, time, numpy as np, io, contextlib
sys.path.insert(0,'/home/arclab/Documents/Emma/git/thread_reconstruction/src')
import matplotlib; matplotlib.use('Agg')
from thread_reconstruction.optim import Optim
class A: pass
O=Optim(A())
fx=fy=800.0;cx,cy=320.0,240.0
cam=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]]);P1=np.hstack([cam,np.zeros((3,1))]);P2=P1.copy()
mask=np.zeros((480,640),np.uint8)
def make(N,seed):
    rng=np.random.default_rng(seed);s=np.linspace(0,1,N)
    return np.column_stack([240+150*np.sin(2*np.pi*s)+rng.normal(0,2,N),
        80+480*s+rng.normal(0,2,N),60+5*np.sin(3*np.pi*s)+rng.normal(0,2,N)]),np.arange(N)
for N in (120,200,300):
    kp,od=make(N,1)
    with contextlib.redirect_stdout(io.StringIO()):
        th,sp=O.optim(mask,kp.copy(),od.copy(),cam,P1,P2,speedy=True)
    ts=[]
    for _ in range(4):
        t0=time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            O.optim(mask,kp.copy(),od.copy(),cam,P1,P2,speedy=True)
        ts.append((time.perf_counter()-t0)*1e3)
    ok = th is not None and np.all(np.isfinite(th['thread'](np.linspace(0,1,50))))
    print(f"N={N}: {min(ts):.0f}ms  ok={ok}")
