# SplineEKF — Kalman Filter Formulation

Kalman filter whose state is the control points of a fixed-knot cubic spline
representing the thread in the 3‑D camera frame. Implemented in
[`spline_ekf.py`](../src/thread_reconstruction/spline_ekf.py).

The observation model is **linear** in the state (cardinal‑basis interpolation),
so the core filter is a plain linear KF. The "EKF" name comes only from the
image‑frame gate, which linearizes the pinhole projection.

---

## 1. State

$M$ control points $p_j \in \mathbb{R}^3$ stacked into a single vector:

$$
\mathbf{x} = \big[\, p_0^\top,\; p_1^\top,\; \dots,\; p_{M-1}^\top \,\big]^\top \in \mathbb{R}^{3M},
\qquad
\mathbf{P} \in \mathbb{R}^{3M \times 3M}.
$$

The knots are fixed for the filter lifetime:

$$
\tau_j = \frac{j}{M-1}, \qquad j = 0, \dots, M-1 .
$$

## 2. Observation model (linear)

The spline is a cardinal‑basis interpolation of the control points,

$$
s(t) = \sum_{j=0}^{M-1} c_j(t)\, p_j,
\qquad c_j(\tau_k) = \delta_{jk},
$$

where each $c_j$ is a natural cubic spline that is $1$ at its own knot and $0$ at
all others (precomputed once). Evaluating the spline at parameter $t$ is linear
in $\mathbf{x}$:

$$
s(t) = H(t)\,\mathbf{x},
\qquad
H(t) = \big[\, c_0(t) I_3,\; c_1(t) I_3,\; \dots,\; c_{M-1}(t) I_3 \,\big] \in \mathbb{R}^{3 \times 3M}.
$$

For $N$ observations at parameters $t_1,\dots,t_N$ the rows stack into
$H \in \mathbb{R}^{3N \times 3M}$.

## 3. Initialization

Seed from a warm spline callable sampled at the knots, with isotropic prior
covariance:

$$
\mathbf{x}_0 = \operatorname{vec}\big(\text{warm}(\tau)\big),
\qquad
\mathbf{P}_0 = \sigma_0^2\, I_{3M}
\quad (\sigma_0^2 = \texttt{P0\_scale}).
$$

## 4. Predict step

Driven by the relative tool motion between frames,
$T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$ (with $R\in SO(3)$, $t\in\mathbb{R}^3$),
and the current tool‑tip position $g \in \mathbb{R}^3$.

### 4.1 Distance‑weighted rigid warp (mean)

Each control point follows the tool's **full translation**, but only the
**near‑tool** portion follows its rotation (so a large tool rotation does not
fling the far, anchored end of the thread):

$$
\delta_j = p_j - g,
\qquad
w_j = \exp\!\left(-\frac{\lVert \delta_j \rVert}{r_d}\right),
$$

$$
\boxed{\;p_j^- = p_j + t + w_j\,(R - I_3)\,\delta_j\;}
$$

where $r_d = \texttt{deform\_radius}$. At $w_j = 1$ (at the tip) this is the full
rigid motion $R\delta_j + g + t$; at $w_j = 0$ (far away) it is pure translation
$p_j + t$.

### 4.2 Transition Jacobian

Block‑diagonal, consistent with the weighted warp:

$$
F = \operatorname{blockdiag}(F_{00}, \dots, F_{M-1,M-1}),
\qquad
F_{jj} = I_3 + w_j\,(R - I_3) \in \mathbb{R}^{3\times 3}.
$$

### 4.3 Motion‑scaled process noise ("temporal memory")

The thread only deforms when the tool moves it, so the whole process‑noise
matrix is scaled by how much the tool actually moved this frame:

$$
\lVert t \rVert,
\qquad
\theta = \arccos\!\left(\frac{\operatorname{tr}(R) - 1}{2}\right),
$$

$$
\text{motion} = \min\!\left(1,\; \frac{\lVert t \rVert}{\tau_t} + \frac{\theta}{\tau_r}\right),
\qquad
q_{\text{scale}} = q_{\text{floor}} + (1 - q_{\text{floor}})\,\text{motion},
$$

with $\tau_t = \texttt{motion\_trans\_ref}$, $\tau_r = \texttt{motion\_rot\_ref}$,
$q_{\text{floor}} = \texttt{q\_motion\_floor}$. When the tool is stationary,
$q_{\text{scale}} \to q_{\text{floor}}$, the Kalman gain shrinks, and the estimate
holds its prior instead of chasing measurement noise.

### 4.4 Spatially‑varying process noise

Points near the grasped tool deform most; the far/anchored end moves rigidly:

$$
\alpha_j = \exp\!\left(-\frac{\lVert p_j^- - g \rVert}{r_d}\right),
\qquad
\sigma_j = \sigma_{\text{base}} + (\sigma_{\text{tip}} - \sigma_{\text{base}})\,\alpha_j,
$$

$$
Q = q_{\text{scale}} \cdot \operatorname{blockdiag}\big(\sigma_0^2 I_3,\; \dots,\; \sigma_{M-1}^2 I_3\big).
$$

### 4.5 Propagation

$$
\boxed{\;\mathbf{x} \leftarrow \mathbf{x}^-, \qquad \mathbf{P} \leftarrow F\,\mathbf{P}\,F^\top + Q\;}
$$

## 5. Update step (information form)

Given $N$ matched keypoints, each a parameter–position pair $(t_i,\, z_i)$ with
$z_i \in \mathbb{R}^3$. Stack $H \in \mathbb{R}^{3N\times 3M}$ and
$\mathbf{z} = \operatorname{vec}(z_i) \in \mathbb{R}^{3N}$.

Per‑observation measurement noise is anisotropic — tight laterally, loose in
depth (stereo depth is noisier):

$$
R_{\text{block}} = \operatorname{diag}(\sigma_m^2,\; \sigma_m^2,\; \sigma_{mz}^2),
\qquad
R = I_N \otimes R_{\text{block}}.
$$

Innovation:

$$
\nu = \mathbf{z} - H\mathbf{x}.
$$

The update is computed in **information form**, which solves a
$3M \times 3M$ system instead of the classic $3N \times 3N$ innovation system
(much smaller, since $N \gg M$):

$$
\boxed{\;
\mathbf{P}^+ = \big(\mathbf{P}^{-1} + H^\top R^{-1} H\big)^{-1},
\qquad
\mathbf{x}^+ = \mathbf{x} + \mathbf{P}^+ H^\top R^{-1} \nu
\;}
$$

followed by symmetrization $\mathbf{P}^+ \leftarrow \tfrac12(\mathbf{P}^+ + \mathbf{P}^{+\top})$.
Because $R$ is block‑diagonal, $R^{-1} = I_N \otimes R_{\text{block}}^{-1}$ is trivial.

This is algebraically identical to the standard Kalman form

$$
S = H\mathbf{P}H^\top + R, \quad
K = \mathbf{P}H^\top S^{-1}, \quad
\mathbf{x}^+ = \mathbf{x} + K\nu, \quad
\mathbf{P}^+ = (I - KH)\mathbf{P},
$$

but avoids forming and inverting the large $3N\times 3N$ matrix $S$.

## 6. Gating

Candidate keypoints are gated before they are accepted as observations. A
candidate is accepted iff its squared Mahalanobis distance is below a
$\chi^2$ threshold.

### 6.1 3‑D gate

For a candidate at parameter $t_i$ with measured 3‑D position $y_i$:

$$
\mu_i = H(t_i)\,\mathbf{x},
\qquad
S_i = H(t_i)\,\mathbf{P}\,H(t_i)^\top + R_{\text{block}},
$$

$$
d_i^2 = (y_i - \mu_i)^\top S_i^{-1} (y_i - \mu_i),
\qquad
\text{accept if } d_i^2 < \chi^2_{3}.
$$

with $\chi^2_3 = \texttt{chi2\_thresh}$ (e.g. $7.81$ at 95 %, 3 DOF).

### 6.2 Image‑frame gate (linearized projection)

To judge a candidate on **where it sits in the image** rather than on noisy
stereo depth, the predicted point and its covariance are pushed through the
pinhole projection. With $\mu = H(t_i)\mathbf{x} = (X, Y, Z)$ and
$S_3 = H(t_i)\mathbf{P}H(t_i)^\top + R_{\text{block}}$:

$$
\pi(X,Y,Z) = \begin{bmatrix} f_y\, Y/Z + c_y \\ f_x\, X/Z + c_x \end{bmatrix}
\quad (\text{row, col}),
$$

$$
J = \frac{\partial \pi}{\partial (X,Y,Z)}
= \begin{bmatrix} 0 & f_y/Z & -f_y\,Y/Z^2 \\ f_x/Z & 0 & -f_x\,X/Z^2 \end{bmatrix},
\qquad
S_2 = J\,S_3\,J^\top \in \mathbb{R}^{2\times 2}.
$$

For a candidate at pixel $u_i$:

$$
d_i^2 = \big(u_i - \pi(\mu)\big)^\top S_2^{-1} \big(u_i - \pi(\mu)\big),
\qquad
\text{accept if } d_i^2 < \chi^2_{2},
$$

with $\chi^2_2 = 5.99$ (95 %, 2 DOF). Points behind the camera ($Z \le 0$) are
rejected.

## 7. Covariance inflation (`degrade`)

Before an update in an ambiguous frame, the covariance is inflated by an
ambiguity score $a \in [0,1]$ so the following update leans on data rather than
the stale prior:

$$
\mathbf{P} \leftarrow (1 + \gamma\, a)\,\mathbf{P},
$$

rescaled if necessary so $\max_i \mathbf{P}_{ii} \le P_{\max}$
($\gamma = \texttt{gain}$, $P_{\max} = \texttt{max\_diag}$).

## 8. Per‑frame cycle

$$
\text{predict}(T, g)
\;\longrightarrow\;
\text{gate candidates}\ (\S 6)
\;\longrightarrow\;
\text{order / match}
\;\longrightarrow\;
\text{update with accepted } (t_i, z_i)\ (\S 5).
$$

## Symbol / parameter glossary

| Symbol | Code | Meaning |
|---|---|---|
| $M$ | `n_ctrl` | number of control points (state dim $= 3M$) |
| $\sigma_m$ | `sigma_meas` | lateral (x, y) measurement noise std |
| $\sigma_{mz}$ | `sigma_meas_z` | depth (z) measurement noise std (default $4\sigma_m$) |
| $\sigma_{\text{base}}$ | `sigma_proc_base` | process noise far from tool (rigid) |
| $\sigma_{\text{tip}}$ | `sigma_proc_tip` | process noise at the tool tip (elastic) |
| $r_d$ | `deform_radius` | e‑folding distance of tool‑induced deformation |
| $\chi^2_3$ | `chi2_thresh` | 3‑D gate threshold (7.81 ≈ 95 %) |
| $\tau_t$ | `motion_trans_ref` | translation counted as "full" motion |
| $\tau_r$ | `motion_rot_ref` | rotation (rad) counted as "full" motion |
| $q_{\text{floor}}$ | `q_motion_floor` | fraction of $Q$ kept when the tool is stationary |
| $\sigma_0^2$ | `P0_scale` | initial per‑coordinate prior variance |
| $f_x, f_y, c_x, c_y$ | `P1` | left‑camera intrinsics (rectified) |
