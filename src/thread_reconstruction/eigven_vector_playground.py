import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import RANSACRegressor

# --- 1. Parameters ---
n = 10                 # Number of points per axis
intersection_radius = 8.0 
pca_color = 'red'       
ransac_color = 'orange'

# --- 2. Randomly Initialize Centroid, Axes, and Variances ---
centroid = np.random.uniform(20, 80, size=2)

# Generate two completely random, independent axes (NOT orthogonal)
axis1 = np.random.uniform(-1, 1, size=2)
axis1 /= np.linalg.norm(axis1)

axis2 = np.random.uniform(-1, 1, size=2)
axis2 /= np.linalg.norm(axis2)

# Generate random standard deviations (variances) for each axis
std1 = np.random.uniform(4.0, 10.0)
std2 = np.random.uniform(4.0, 10.0)

# --- 3. Generate n Points Along Each Axis ---
dist1 = np.random.normal(0, std1, n)
pts1 = centroid + np.outer(dist1, axis1)

dist2 = np.random.normal(0, std2, n)
pts2 = centroid + np.outer(dist2, axis2)

neighbour_pts = np.vstack((pts1, pts2))

# Add a tiny bit of uniform 2D scatter noise
neighbour_pts += np.random.normal(0, 0.6, neighbour_pts.shape)

# --- 4. Setup Plot ---
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(neighbour_pts[:, 1], neighbour_pts[:, 0], alpha=0.3, color='gray', label='Generated Points')
ax.scatter(centroid[1], centroid[0], color='black', marker='x', s=100, label='Centroid', zorder=5)

# Plot Ground Truth
ax.plot([centroid[1] - axis1[1]*std1*2, centroid[1] + axis1[1]*std1*2], 
        [centroid[0] - axis1[0]*std1*2, centroid[0] + axis1[0]*std1*2], 
        color='blue', linestyle='--', linewidth=2, label='True Random Axis 1')
ax.plot([centroid[1] - axis2[1]*std2*2, centroid[1] + axis2[1]*std2*2], 
        [centroid[0] - axis2[0]*std2*2, centroid[0] + axis2[0]*std2*2], 
        color='green', linestyle='--', linewidth=2, label='True Random Axis 2')

# ==========================================
# --- 5. YOUR PCA SNIPPET (Red Arrows) ---
# ==========================================
unit_vecs_nb = neighbour_pts - centroid
norms_nb = np.linalg.norm(unit_vecs_nb, axis=1, keepdims=True)
norms_nb = np.where(norms_nb < 1e-6, 1.0, norms_nb)
uv = unit_vecs_nb / norms_nb
cov = uv.T @ uv / len(uv)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]

for ev_i in range(2):
    scale = intersection_radius * 0.8 * (eigvals[ev_i] / (eigvals[0] + 1e-8)) ** 0.5
    ev = eigvecs[:, ev_i]  # (y, x) in image space
    ax.annotate(
        "",
        xy=(centroid[1] + ev[1] * scale, centroid[0] + ev[0] * scale),
        xytext=(centroid[1] - ev[1] * scale, centroid[0] - ev[0] * scale),
        arrowprops=dict(arrowstyle="<->", color=pca_color, lw=2.5 if ev_i == 0 else 1.5),
    )

# ==========================================
# --- 6. ITERATIVE RANSAC (Orange Arrows) ---
# ==========================================
# Prepare data for RANSAC (X must be 2D, y must be 1D)
X_data = neighbour_pts[:, 1].reshape(-1, 1) # X-coordinates
y_data = neighbour_pts[:, 0]                # Y-coordinates

ransac_vectors = []

for i in range(2):
    if len(X_data) < 2:
        break
        
    # Fit RANSAC to the current points
    ransac = RANSACRegressor(residual_threshold=2.0, random_state=42)
    ransac.fit(X_data, y_data)
    
    # Identify inliers (the points belonging to this specific line)
    inlier_mask = ransac.inlier_mask_
    
    # Calculate the exact axis vector of JUST these inliers using local covariance
    # (We do this instead of using the line's slope to avoid math errors if a line is perfectly vertical)
    inlier_X = X_data[inlier_mask].flatten()
    inlier_y = y_data[inlier_mask]
    inlier_pts = np.column_stack((inlier_y, inlier_X)) # Convert back to [y, x]
    
    # Simple PCA on just the isolated inliers to get the principal direction vector
    centered_inliers = inlier_pts - inlier_pts.mean(axis=0)
    inlier_cov = centered_inliers.T @ centered_inliers
    inlier_eigvals, inlier_eigvecs = np.linalg.eigh(inlier_cov)
    
    # Store the primary eigenvector (the direction of this specific arm)
    ransac_vectors.append(inlier_eigvecs[:, np.argmax(inlier_eigvals)])
    
    # Remove the inliers from the dataset so the next loop finds the *other* axis
    X_data = X_data[~inlier_mask]
    y_data = y_data[~inlier_mask]

# Plot the RANSAC vectors
for ev in ransac_vectors:
    scale = intersection_radius * 0.8
    ax.annotate(
        "",
        xy=(centroid[1] + ev[1] * scale, centroid[0] + ev[0] * scale),
        xytext=(centroid[1] - ev[1] * scale, centroid[0] - ev[0] * scale),
        arrowprops=dict(arrowstyle="<->", color=ransac_color, lw=2.5),
    )

# --- 7. Finalize Plot Details ---
ax.set_aspect('equal')
ax.set_title("Comparison: PCA (Orthogonal) vs Iterative RANSAC (Geometric)")
ax.set_xlabel("X Axis (index 1)")
ax.set_ylabel("Y Axis (index 0)")

# Custom Legend
custom_lines = [
    Line2D([0], [0], color='blue', linestyle='--', lw=2),
    Line2D([0], [0], color='green', linestyle='--', lw=2),
    Line2D([0], [0], color=pca_color, lw=2.5, marker='<', markersize=8),
    Line2D([0], [0], color=ransac_color, lw=2.5, marker='<', markersize=8)
]
ax.legend(custom_lines, ['True Axis 1', 'True Axis 2', 'PCA (Forced 90°)', 'RANSAC (Line Fitting)'], loc='upper right')

plt.grid(True, linestyle=':', alpha=0.7)
plt.show()