import numpy as np
import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)

    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-8)

    u = np.cross(r, f)
    u = u / (np.linalg.norm(u) + 1e-8)

    R = np.stack([r, u, -f], axis=1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def rot_z(deg_clockwise: float):
    """Rotation about +Z. Clockwise in screen coords corresponds to negative angle."""
    a = np.deg2rad(-deg_clockwise)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def rot_x(deg: float):
    """Rotation about +X (right axis). 180 deg flips up/down."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c,  -s],
                     [0.0, s,   c]], dtype=np.float32)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()
    config.flame_model_path = "./model/generic_model.pkl"
    config.static_landmark_embedding_path = "./model/flame_static_embedding.pkl"
    config.dynamic_landmark_embedding_path = "./model/flame_dynamic_embedding.npy"

    B = 1
    if hasattr(config, "batch_size"):
        config.batch_size = B

    flamelayer = FLAME(config).to(device)

    # neutral, front pose
    shape_params = torch.zeros(B, 100, dtype=torch.float32, device=device)
    expression_params = torch.zeros(B, 50, dtype=torch.float32, device=device)
    pose_params = torch.zeros(B, 6, dtype=torch.float32, device=device)

    neck_pose = None
    eye_pose = None
    use_neck_eye = bool(getattr(config, "optimize_neckpose", False)) and bool(
        getattr(config, "optimize_eyeballpose", False)
    )
    if use_neck_eye:
        neck_pose = torch.zeros(B, 3, dtype=torch.float32, device=device)
        eye_pose = torch.zeros(B, 6, dtype=torch.float32, device=device)

    if neck_pose is None:
        vertice, _landmark = flamelayer(shape_params, expression_params, pose_params)
    else:
        vertice, _landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    vertices = vertice[0].detach().cpu().numpy()
    faces = flamelayer.faces

    # 1) Center mesh
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = vertices - center

    # 2) Rotate in image plane (your original)
    #Rz = rot_z(180.0)
    Rz = rot_z(0.0)
    vertices = (Rz @ vertices.T).T

    # 3) FIX: flip vertically so head is up, shoulders down
    # (this corrects the upside-down result you showed)
    #Rx = rot_x(180.0)
    Rx = rot_x(0.0)
    vertices = (Rx @ vertices.T).T

    # 4) Camera distance to fit
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    radius = 0.5 * np.linalg.norm(extent) + 1e-6

    yfov_deg = 25.0
    yfov = np.deg2rad(yfov_deg)
    padding = 1.35
    dist = (radius / np.tan(yfov / 2.0)) * padding

    camera = pyrender.PerspectiveCamera(yfov=yfov)

    cam_pose = look_at(
        eye=(0.0, 0.0, dist),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )

    # Build mesh (NO landmark spheres)
    vertex_colors = np.ones((vertices.shape[0], 4), dtype=np.float32) * np.array(
        [0.78, 0.65, 0.58, 1.0], dtype=np.float32
    )
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[0.35, 0.35, 0.35],
    )
    scene.add(render_mesh)

    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=cam_pose)

    pyrender.Viewer(
        scene,
        use_raymond_lighting=False,
        viewport_size=(1200, 900),
        camera=camera,
        camera_pose=cam_pose,
    )


if __name__ == "__main__":
    main()