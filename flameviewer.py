import numpy as np
import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Config + model asset paths
    # -------------------------
    config = get_config()
    config.flame_model_path = "./model/generic_model.pkl"
    config.static_landmark_embedding_path = "./model/flame_static_embedding.pkl"
    config.dynamic_landmark_embedding_path = "./model/flame_dynamic_embedding.npy"

    # -------------------------
    # IMPORTANT: keep batch size consistent everywhere
    # Some FLAME implementations store config.batch_size (=8 by default).
    # Force it to match your actual batch (B).
    # -------------------------
    B = 1
    if hasattr(config, "batch_size"):
        config.batch_size = B

    print("creating the FLAME Decoder")
    flamelayer = FLAME(config).to(device)

    rad = np.pi / 180.0

    shape_params = torch.zeros(B, 100, dtype=torch.float32, device=device)       # (B,100)
    expression_params = torch.zeros(B, 50, dtype=torch.float32, device=device)   # (B,50)
    pose_params = torch.tensor(
        [[0.0, 10.0 * rad, 0.0, 0.0, 0.0, 0.0]],  # (B,6)
        dtype=torch.float32,
        device=device,
    )

    neck_pose = None
    eye_pose = None
    use_neck_eye = bool(getattr(config, "optimize_neckpose", False)) and bool(
        getattr(config, "optimize_eyeballpose", False)
    )
    if use_neck_eye:
        neck_pose = torch.zeros(B, 3, dtype=torch.float32, device=device)  # (B,3)
        eye_pose = torch.zeros(B, 6, dtype=torch.float32, device=device)   # (B,6)

    # Debug prints (this is the FIRST thing to check when batch errors happen)
    print("shape_params:", tuple(shape_params.shape))
    print("expression_params:", tuple(expression_params.shape))
    print("pose_params:", tuple(pose_params.shape))
    if neck_pose is not None:
        print("neck_pose:", tuple(neck_pose.shape))
        print("eye_pose:", tuple(eye_pose.shape))

    # Forward
    if neck_pose is None:
        vertice, landmark = flamelayer(shape_params, expression_params, pose_params)
    else:
        vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    print("Vertices:", vertice.shape)
    print("Landmarks:", landmark.shape)

    vertices = vertice[0].detach().cpu().numpy()
    joints = landmark[0].detach().cpu().numpy()
    faces = flamelayer.faces

    # Render
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.78, 0.65, 0.58, 1.0]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh)

    sm = trimesh.creation.uv_sphere(radius=0.004)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_mesh)

    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    main()