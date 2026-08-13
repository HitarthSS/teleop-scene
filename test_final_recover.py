import sys, numpy as np, io, contextlib
sys.path.insert(0,'/home/arclab/Documents/Emma/git/thread_reconstruction/src')
import matplotlib; matplotlib.use('Agg')
from thread_reconstruction.spline_ekf import SplineEKF
from scipy.interpolate import CubicSpline
M=20; s=np.linspace(0,1,M)
gt=CubicSpline(s,np.column_stack([80*s,20*np.sin(np.pi*s),130+4*s]))
fx=fy=800.;cx,cy=320.,240.; P1=np.array([[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0.]])
def to_px(p3):
    pr=(P1@np.c_[p3,np.ones(len(p3))].T).T; pr/=pr[:,2:3]+1e-7; return pr[:,[1,0]]
def scen(corrupt):
    e=SplineEKF(n_ctrl=M,sigma_meas=4.0,sigma_meas_z=10.0,sigma_proc_base=0.5,
       sigma_proc_tip=6.0,deform_radius=50.0,chi2_thresh=6.25,sigma_smooth=10.0,
       sigma_stretch=0.99,sigma_end_straight=4.0,end_span=3)  # all defaults incl recovery
    rng=np.random.default_rng(0);N=90;t_dense=np.linspace(0,1,500);errs=[];nrec=0
    with contextlib.redirect_stdout(io.StringIO()):
        e.initialize(gt,P0_scale=9.0)
        for f in range(30):
            manip=corrupt and 5<=f<10; trans=np.eye(4)
            if manip: trans[:3,3]=[8.,0.,0.]
            e.predict(trans,np.array([0.,0.,130.]))
            mt=np.sort(rng.uniform(0,1,N)); base=np.asarray(gt(mt))
            if manip: base+=np.array([60.,40.,0.])
            mz=base+np.column_stack([rng.normal(0,1.5,N),rng.normal(0,1.5,N),rng.normal(0,3,N)])
            kpx=to_px(mz); idx=np.clip(np.round(mt*(len(t_dense)-1)).astype(int),0,len(t_dense)-1)
            if e.maybe_recover_from_divergence(): nrec+=1
            mask,_=e.mahalanobis_gate_2d(kpx,idx,t_dense,P1)
            if mask.sum()>=4: e.update(mt[mask],mz[mask])
            p=np.asarray(e.get_spline()(np.linspace(0,1,50)));g=np.asarray(gt(np.linspace(0,1,50)))
            errs.append(np.linalg.norm(p-g,axis=1).mean())
    return errs,nrec
e_c,n_c=scen(True); e_g,n_g=scen(False)
print(f"CORRUPTED run: err peak(f9)={e_c[9]:.1f} -> recovered f29={e_c[29]:.2f}  (recovery fired {n_c}x)")
print(f"CLEAN run:     err max={max(e_g):.2f} f29={e_g[29]:.2f}  (recovery fired {n_g}x  <- must be 0)")
