"""
Mac-compatible FLAME demo
Fixed version of original FLAME PyTorch example

Changes:
 - Removed CUDA calls
 - Added MPS / CPU device selection
 - Added explicit model path setup
"""

import numpy as np
import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config

# ------------------------------------------------------------
# Device selection (Mac compatible)
# ------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------
config = get_config()

# IMPORTANT: Set correct local paths
config.flame_model_path = "./model/generic_model.pkl"
config.static_landmark_embedding_path = "./model/flame_static_embedding.pkl"
config.dynamic_landmark_embedding_path = "./model/flame_dynamic_embedding.npy"

# ------------------------------------------------------------
# Create FLAME model
# ------------------------------------------------------------
radian = np.pi / 180.0
flamelayer = FLAME(config).to(device)

# ------------------------------------------------------------
# Create parameters
# ------------------------------------------------------------

# Mean shape batch
shape_params = torch.zeros(8, 100, device=device)

# Global + jaw poses
pose_params_numpy = np.array(
    [
        [0.0, 30.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -30.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, 85.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -48.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, 10.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, -15.0 * radian, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

pose_params = torch.tensor(pose_params_numpy, device=device)

# Neutral expressions
expression_params = torch.zeros(8, 50, device=device)

# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------
vertice, landmark = flamelayer(
    shape_params,
    expression_params,
    pose_params
)

print("Vertices:", vertice.size())
print("Landmarks:", landmark.size())

# Optional neck/eye pose
if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(8, 3, device=device)
    eye_pose = torch.zeros(8, 6, device=device)
    vertice, landmark = flamelayer(
        shape_params,
        expression_params,
        pose_params,
        neck_pose,
        eye_pose,
    )

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
faces = flamelayer.faces

for i in range(8):
    vertices = vertice[i].detach().cpu().numpy()
    joints = landmark[i].detach().cpu().numpy()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)

    # Draw landmark spheres
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]

    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints

    joints_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_mesh)

    pyrender.Viewer(scene, use_raymond_lighting=True)