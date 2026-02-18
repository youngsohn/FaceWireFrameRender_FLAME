"""
Front-view FLAME "photo paint" (NO shape/pose fitting)
- Keeps FLAME geometry unchanged (shape=0, expr=0, pose=0)
- Detects 468 FaceMesh landmarks, selects 68 subset
- Computes a best-fit linear projection Q ~= A*P + t (least squares)
  where P are FLAME 3D landmarks, Q are image 2D landmarks.
- Uses that projection to sample photo colors -> vertex colors
- Colors ONLY front-facing vertices (simple visibility heuristic)
- Renders with pyrender

Files written:
- debug_landmarks.png       : photo + detected 68 landmarks (green)
- debug_fit_overlay.png     : photo + detected (green) + projected FLAME lmk (red)
"""

import os
import cv2
import numpy as np
import torch
import trimesh
import pyrender
import mediapipe as mp

from flame_pytorch import FLAME, get_config


# --------------------------
# USER SETTINGS
# --------------------------
IMAGE_PATH = "face1.png"  # your input face photo
OUT_DEBUG_LMK = "debug_landmarks.png"
OUT_DEBUG_OVERLAY = "debug_fit_overlay.png"

# If your FLAME mesh comes out rotated, tweak these 3 until upright
ROT_X_DEG = -90.0
ROT_Y_DEG = 0.0
ROT_Z_DEG = 0.0

# FaceMesh -> 68 indices (approx). You already used this list.
LANDMARK_POINTS_68 = [
    162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
    71,63,105,66,107,336,296,334,293,301,
    168,197,5,4,75,97,2,326,305,
    33,160,158,133,153,144,362,385,387,263,373,380,
    61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
]

# “neutral” color for vertices that we don't paint
NEUTRAL_RGB = np.array([0.78, 0.65, 0.58], dtype=np.float32)  # (R,G,B) in [0..1]

# Render controls
VIEWPORT_W, VIEWPORT_H = 1200, 900
YFOV_DEG = 22.0


# --------------------------
# small math helpers
# --------------------------
def rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float32)

def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)

def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

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

    # pyrender camera looks down -Z in camera local coords
    R = np.stack([r, u, -f], axis=1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


# --------------------------
# 1) MediaPipe landmarks (68 subset)
# --------------------------
def detect_landmarks_68(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        res = face_mesh.process(img_rgb)

    if not res.multi_face_landmarks:
        raise RuntimeError("No face detected by MediaPipe FaceMesh.")

    lm = res.multi_face_landmarks[0].landmark
    pts468 = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    pts68 = pts468[LANDMARK_POINTS_68].copy()  # (68,2)
    return pts68


# --------------------------
# 2) Build FLAME neutral mesh + FLAME 3D landmarks (NO fitting)
# --------------------------
def flame_neutral_mesh_and_landmarks(device):
    config = get_config()
    config.flame_model_path = "./model/generic_model.pkl"
    config.static_landmark_embedding_path = "./model/flame_static_embedding.pkl"
    config.dynamic_landmark_embedding_path = "./model/flame_dynamic_embedding.npy"

    # batch=1
    if hasattr(config, "batch_size"):
        config.batch_size = 1

    flamelayer = FLAME(config).to(device)

    shape = torch.zeros((1, 100), dtype=torch.float32, device=device)
    expr  = torch.zeros((1, 50),  dtype=torch.float32, device=device)
    pose  = torch.zeros((1, 6),   dtype=torch.float32, device=device)

    # some impls have neck/eye enabled; pass zeros if required
    neck_pose = None
    eye_pose = None
    use_neck_eye = bool(getattr(config, "optimize_neckpose", False)) and bool(getattr(config, "optimize_eyeballpose", False))
    if use_neck_eye:
        neck_pose = torch.zeros((1, 3), dtype=torch.float32, device=device)
        eye_pose  = torch.zeros((1, 6), dtype=torch.float32, device=device)

    with torch.no_grad():
        if neck_pose is None:
            verts, lmk3d = flamelayer(shape, expr, pose)
        else:
            verts, lmk3d = flamelayer(shape, expr, pose, neck_pose, eye_pose)

    vertices = verts[0].detach().cpu().numpy().astype(np.float32)   # (N,3)
    lmk3d_68 = lmk3d[0].detach().cpu().numpy().astype(np.float32)   # (68,3)
    faces = flamelayer.faces

    return vertices, faces, lmk3d_68


# --------------------------
# 3) Compute best-fit 3D->2D map: q ~= A*p + t  (least squares)
# --------------------------
def solve_affine_3d_to_2d(P3, Q2):
    """
    P3: (K,3) FLAME 3D landmarks (after same centering/rotation as vertices)
    Q2: (K,2) image 2D landmarks in pixels
    Solve least squares:
        Q ≈ P @ A^T + t
    Returns A (2,3), t (2,)
    """
    K = P3.shape[0]
    assert Q2.shape[0] == K

    # augment P with 1 for translation: [x y z 1]
    X = np.concatenate([P3, np.ones((K, 1), dtype=np.float32)], axis=1)  # (K,4)

    # solve X @ B ≈ Q, where B is (4,2)
    B, *_ = np.linalg.lstsq(X, Q2, rcond=None)  # B: (4,2)
    B = B.astype(np.float32)

    A = B[:3, :].T         # (2,3)
    t = B[3, :].reshape(2) # (2,)
    return A, t


# --------------------------
# 4) Paint vertices using the projection, only front-facing
# --------------------------
def sample_vertex_colors(img_bgr, V, F, A, t):
    """
    img_bgr: HxWx3
    V: (N,3) vertices in same space used to compute A,t
    F: (M,3) faces
    A: (2,3)
    t: (2,)
    """
    h, w = img_bgr.shape[:2]

    # Compute vertex normals for front-facing test
    tri = trimesh.Trimesh(V, F, process=False)
    tri.fix_normals()
    vn = tri.vertex_normals.astype(np.float32)  # (N,3)

    # Our render camera will be on +Z looking toward origin, so view dir ~ (0,0,-1).
    # "Facing camera" roughly means normal_z > 0 (because dot((0,0,-1), n) < 0  <=> n_z > 0).
    front = vn[:, 2] > 0.0

    # Project
    uv = (A @ V.T).T + t[None, :]   # (N,2)
    x = uv[:, 0]
    y = uv[:, 1]

    # bilinear sample
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = (x >= 0) & (x <= (w - 2)) & (y >= 0) & (y <= (h - 2)) & front

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    Ia = img_bgr[y0c, x0c].astype(np.float32)
    Ib = img_bgr[y1c, x0c].astype(np.float32)
    Ic = img_bgr[y0c, x1c].astype(np.float32)
    Id = img_bgr[y1c, x1c].astype(np.float32)

    col_bgr = (wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id)
    col_rgb = col_bgr[:, ::-1] / 255.0  # to RGB

    # default neutral
    rgb = np.tile(NEUTRAL_RGB[None, :], (V.shape[0], 1)).astype(np.float32)
    rgb[inside] = col_rgb[inside]

    alpha = np.ones((V.shape[0], 1), dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=1)
    return rgba


# --------------------------
# 5) Render
# --------------------------
def render_mesh(V, F, vertex_rgba):
    tri_mesh = trimesh.Trimesh(V, F, process=False)
    tri_mesh.visual.vertex_colors = (vertex_rgba * 255.0).astype(np.uint8)
    tri_mesh.fix_normals()

    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    # auto-fit camera distance
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    extent = vmax - vmin
    radius = 0.5 * np.linalg.norm(extent) + 1e-6

    yfov = np.deg2rad(YFOV_DEG)
    dist = (radius / np.tan(yfov / 2.0)) * 1.35

    cam_pose = look_at(
        eye=(0.0, 0.0, dist),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )
    camera = pyrender.PerspectiveCamera(yfov=yfov)

    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.25, 0.25, 0.25])
    scene.add(mesh)
    scene.add(camera, pose=cam_pose)

    # light from camera
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    pyrender.Viewer(
        scene,
        use_raymond_lighting=False,
        viewport_size=(VIEWPORT_W, VIEWPORT_H),
        camera=camera,
        camera_pose=cam_pose,
    )


# --------------------------
# MAIN
# --------------------------
def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Put your face photo as '{IMAGE_PATH}' or change IMAGE_PATH.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Failed to read image. Check path and file type.")
    h, w = img_bgr.shape[:2]
    print("Image:", IMAGE_PATH, "shape:", img_bgr.shape)

    # 1) photo landmarks
    lm2d = detect_landmarks_68(img_bgr)

    dbg = img_bgr.copy()
    for (x, y) in lm2d:
        cv2.circle(dbg, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(OUT_DEBUG_LMK, dbg)
    print("[OK] wrote", OUT_DEBUG_LMK)

    # 2) FLAME neutral mesh + 3D landmarks
    V0, F, L3 = flame_neutral_mesh_and_landmarks(device)

    # 3) Apply the SAME centering/rotation to BOTH vertices and 3D landmarks
    # center using vertices bbox (stable)
    center = 0.5 * (V0.min(axis=0) + V0.max(axis=0))
    V = V0 - center[None, :]
    L = L3 - center[None, :]

    R_fix = (rot_z(ROT_Z_DEG) @ rot_y(ROT_Y_DEG) @ rot_x(ROT_X_DEG)).astype(np.float32)
    V = (R_fix @ V.T).T
    L = (R_fix @ L.T).T

    # 4) Solve 3D->2D projection WITHOUT changing shape
    A, t = solve_affine_3d_to_2d(L, lm2d)

    # debug overlay: project FLAME landmarks back to image
    proj = (A @ L.T).T + t[None, :]
    vis = img_bgr.copy()
    for (x, y) in lm2d:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)  # green
    for (x, y) in proj:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)  # red
    cv2.imwrite(OUT_DEBUG_OVERLAY, vis)
    print("[OK] wrote", OUT_DEBUG_OVERLAY)

    # 5) Paint vertex colors from photo (front-facing only)
    rgba = sample_vertex_colors(img_bgr, V, F, A, t)

    # 6) Render
    render_mesh(V, F, rgba)


if __name__ == "__main__":
    main()