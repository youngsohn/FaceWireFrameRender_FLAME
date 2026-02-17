import os
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

    # Camera looks down -Z in camera coords
    R = np.stack([r, u, -f], axis=1)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def debug_print_vec(name, v):
    v = np.asarray(v).reshape(-1)
    print(f"{name}: [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]")


def main():
    print("=== FLAME VIEWER DEBUG ===")
    print("cwd:", os.getcwd())

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Config + paths
    # -------------------------
    config = get_config()
    config.flame_model_path = "./model/generic_model.pkl"
    config.static_landmark_embedding_path = "./model/flame_static_embedding.pkl"
    config.dynamic_landmark_embedding_path = "./model/flame_dynamic_embedding.npy"

    B = 1
    if hasattr(config, "batch_size"):
        print("config.batch_size (before):", getattr(config, "batch_size"))
        config.batch_size = B
        print("config.batch_size (after):", getattr(config, "batch_size"))

    print("creating the FLAME Decoder")
    flamelayer = FLAME(config).to(device)

    # -------------------------
    # Neutral params, FRONT VIEW
    # -------------------------
    shape_params = torch.zeros(B, 100, dtype=torch.float32, device=device)
    expression_params = torch.zeros(B, 50, dtype=torch.float32, device=device)
    pose_params = torch.zeros(B, 6, dtype=torch.float32, device=device)

    neck_pose = None
    eye_pose = None
    use_neck_eye = bool(getattr(config, "optimize_neckpose", False)) and bool(
        getattr(config, "optimize_eyeballpose", False)
    )
    print("use_neck_eye:", use_neck_eye)
    if use_neck_eye:
        neck_pose = torch.zeros(B, 3, dtype=torch.float32, device=device)
        eye_pose = torch.zeros(B, 6, dtype=torch.float32, device=device)

    # Forward
    if neck_pose is None:
        vertice, _landmark = flamelayer(shape_params, expression_params, pose_params)
    else:
        vertice, _landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    vertices = vertice[0].detach().cpu().numpy()
    faces = flamelayer.faces

    print("vertices shape:", vertices.shape)
    print("faces shape:", faces.shape)

    # -------------------------
    # Center mesh (bbox center)
    # -------------------------
    vmin0 = vertices.min(axis=0)
    vmax0 = vertices.max(axis=0)
    center0 = 0.5 * (vmin0 + vmax0)
    extent0 = vmax0 - vmin0

    print("--- raw mesh bbox ---")
    debug_print_vec("vmin0", vmin0)
    debug_print_vec("vmax0", vmax0)
    debug_print_vec("center0", center0)
    debug_print_vec("extent0", extent0)

    vertices = vertices - center0

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = vmax - vmin

    print("--- centered mesh bbox (should be near 0) ---")
    debug_print_vec("vmin", vmin)
    debug_print_vec("vmax", vmax)
    debug_print_vec("center", center)
    debug_print_vec("extent", extent)

    # -------------------------
    # Camera fit (front view)
    # -------------------------
    radius = 0.5 * np.linalg.norm(extent) + 1e-6

    yfov_deg = 25.0
    yfov = np.deg2rad(yfov_deg)
    padding = 1.45
    dist = (radius / np.tan(yfov / 2.0)) * padding

    print("--- camera params ---")
    print("radius:", float(radius))
    print("yfov_deg:", yfov_deg)
    print("padding:", padding)
    print("dist:", float(dist))

    eye = np.array([0.0, 0.0, dist], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_pose = look_at(eye, target=target, up=(0.0, 1.0, 0.0))

    debug_print_vec("eye", eye)
    debug_print_vec("target", target)
    print("cam_pose:\n", cam_pose)

    # -------------------------
    # Mesh + scene
    # -------------------------
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

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    cam_node = scene.add(camera, pose=cam_pose)

    # Try to mark it main camera if supported
    try:
        scene.main_camera_node = cam_node
        print("scene.main_camera_node set")
    except Exception as e:
        print("scene.main_camera_node not supported:", repr(e))

    # Light from camera direction
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=cam_pose)

    # -------------------------
    # Offscreen debug render (authoritative)
    # -------------------------
    out_png = "debug_render.png"
    try:
        r = pyrender.OffscreenRenderer(viewport_width=1200, viewport_height=900)
        color, depth = r.render(scene)
        r.delete()
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio
        imageio.imwrite(out_png, color)
        print(f"[OK] wrote {out_png} (check this file first!)")
    except Exception as e:
        print("[WARN] OffscreenRenderer failed:", repr(e))
        print("Proceeding to Viewer...")

    # -------------------------
    # Viewer (may ignore camera on some setups; offscreen PNG is the truth)
    # -------------------------
    pyrender.Viewer(
        scene,
        use_raymond_lighting=False,
        viewport_size=(1200, 900),
        camera=camera,
        camera_pose=cam_pose,
    )


if __name__ == "__main__":
    main()