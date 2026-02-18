"""
Front-view FLAME "photo paint" (NO shape/pose fitting) — improved mapping
- Keeps FLAME geometry unchanged (shape=0, expr=0, pose=0)
- Detects MediaPipe FaceMesh landmarks
    * 68 subset for 3D->2D alignment
    * ordered face-oval polygon for a robust face mask
- Uses best-fit affine map Q ~= A*P + t to project vertices to image
- Samples photo colors inside face mask; fills outside with estimated skin color
- Renders with pyrender using FLAT shading (no lighting influence)

Files written:
- debug_landmarks.png       : photo + detected 68 landmarks (green) + face-oval (blue)
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
#IMAGE_PATH = "face1.png"
IMAGE_PATH = "face4.png"
OUT_DEBUG_LMK = "debug_landmarks.png"
OUT_DEBUG_OVERLAY = "debug_fit_overlay.png"

ROT_X_DEG = -90.0
ROT_Y_DEG = 0.0
ROT_Z_DEG = 0.0

LANDMARK_POINTS_68 = [
    162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
    71,63,105,66,107,336,296,334,293,301,
    168,197,5,4,75,97,2,326,305,
    33,160,158,133,153,144,362,385,387,263,373,380,
    61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
]

# Ordered face oval indices (this is the critical fix).
# This list is widely used with MediaPipe FaceMesh.
FACE_OVAL_ORDERED = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

# Fallback if skin estimation fails
NEUTRAL_RGB = np.array([0.78, 0.65, 0.58], dtype=np.float32)

# Render controls
VIEWPORT_W, VIEWPORT_H = 1200, 900
YFOV_DEG = 22.0

# ---- knobs you can tune ----
# Reduce overall brightness by 50% => 0.5
BRIGHTNESS = 1.0

# If you want to still use a visibility constraint, set this like 0.0 or -0.2
# For "copy full facial area", you typically want to disable it:
USE_FRONT_FACING_TEST = False
FRONT_NORMAL_Z_THRESH = 0.0

# Sampling mode: "nearest" preserves details better than bilinear with vertex colors
SAMPLE_MODE = "nearest"   # "nearest" or "bilinear"


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

    R = np.stack([r, u, -f], axis=1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


# --------------------------
# 1) MediaPipe landmarks (68 subset) + face mask
# --------------------------
def detect_landmarks_68_and_facemask(img_bgr):
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

    pts68 = pts468[LANDMARK_POINTS_68].copy()

    # ordered face-oval polygon
    oval_pts = pts468[FACE_OVAL_ORDERED].copy()
    poly = np.round(oval_pts).astype(np.int32)

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [poly], 255)

    return pts68, face_mask, oval_pts


def estimate_skin_rgb01(img_bgr, face_mask, lo_pct=15.0, hi_pct=85.0):
    """
    Robust skin color estimate from inside the face mask.
    Removes too-dark / too-bright pixels and takes median.
    """
    m = face_mask > 0
    pix = img_bgr[m]  # (M,3) BGR
    if pix.size == 0:
        return NEUTRAL_RGB.copy()

    p = pix.astype(np.float32)
    y = 0.114 * p[:, 0] + 0.587 * p[:, 1] + 0.299 * p[:, 2]  # luma-ish

    lo = np.percentile(y, lo_pct)
    hi = np.percentile(y, hi_pct)
    keep = (y >= lo) & (y <= hi)
    p2 = p[keep] if np.any(keep) else p

    med_bgr = np.median(p2, axis=0)
    med_rgb01 = (med_bgr[::-1] / 255.0).astype(np.float32)
    return med_rgb01


# --------------------------
# 2) Build FLAME neutral mesh + FLAME 3D landmarks (NO fitting)
# --------------------------
def flame_neutral_mesh_and_landmarks(device):
    config = get_config()
    config.flame_model_path = "./model/generic_model.pkl"
    config.static_landmark_embedding_path = "./model/flame_static_embedding.pkl"
    config.dynamic_landmark_embedding_path = "./model/flame_dynamic_embedding.npy"

    if hasattr(config, "batch_size"):
        config.batch_size = 1

    flamelayer = FLAME(config).to(device)

    shape = torch.zeros((1, 100), dtype=torch.float32, device=device)
    expr  = torch.zeros((1, 50),  dtype=torch.float32, device=device)
    pose  = torch.zeros((1, 6),   dtype=torch.float32, device=device)

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

    vertices = verts[0].detach().cpu().numpy().astype(np.float32)
    lmk3d_68 = lmk3d[0].detach().cpu().numpy().astype(np.float32)
    faces = flamelayer.faces

    return vertices, faces, lmk3d_68


# --------------------------
# 3) Compute best-fit 3D->2D map: q ~= A*p + t
# --------------------------
def solve_affine_3d_to_2d(P3, Q2):
    K = P3.shape[0]
    assert Q2.shape[0] == K

    X = np.concatenate([P3, np.ones((K, 1), dtype=np.float32)], axis=1)  # (K,4)
    B, *_ = np.linalg.lstsq(X, Q2, rcond=None)  # (4,2)
    B = B.astype(np.float32)

    A = B[:3, :].T
    t = B[3, :].reshape(2)
    return A, t


# --------------------------
# 4) Paint vertices using face-mask gated projection
# --------------------------
def sample_vertex_colors(img_bgr, face_mask, V, F, A, t, skin_rgb01):
    h, w = img_bgr.shape[:2]

    # Optional front-facing gating (usually OFF for "copy full facial area")
    if USE_FRONT_FACING_TEST:
        tri = trimesh.Trimesh(V, F, process=False)
        tri.fix_normals()
        vn = tri.vertex_normals.astype(np.float32)
        front = vn[:, 2] > float(FRONT_NORMAL_Z_THRESH)
    else:
        front = np.ones((V.shape[0],), dtype=bool)

    # Project vertices to image
    uv = (A @ V.T).T + t[None, :]
    x = uv[:, 0]
    y = uv[:, 1]

    # Determine which projected points land inside the face mask
    xi = np.round(x).astype(np.int32)
    yi = np.round(y).astype(np.int32)
    inside_img = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & front
    inside_face = np.zeros_like(inside_img, dtype=bool)
    valid = np.where(inside_img)[0]
    inside_face[valid] = (face_mask[yi[valid], xi[valid]] > 0)

    # Start with skin fill everywhere (removes gray head/neck)
    rgb = np.tile(skin_rgb01[None, :], (V.shape[0], 1)).astype(np.float32)

    if SAMPLE_MODE == "nearest":
        idx = np.where(inside_face)[0]
        if idx.size > 0:
            col_bgr = img_bgr[yi[idx], xi[idx]].astype(np.float32)
            rgb[idx] = (col_bgr[:, ::-1] / 255.0)

    elif SAMPLE_MODE == "bilinear":
        # Bilinear for smoother, but can look a bit “blurred”
        # Use only for inside_face points
        x = np.clip(x, 0, w - 2)
        y = np.clip(y, 0, h - 2)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        Ia = img_bgr[y0, x0].astype(np.float32)
        Ib = img_bgr[y1, x0].astype(np.float32)
        Ic = img_bgr[y0, x1].astype(np.float32)
        Id = img_bgr[y1, x1].astype(np.float32)

        col_bgr = (wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id)
        col_rgb = col_bgr[:, ::-1] / 255.0
        rgb[inside_face] = col_rgb[inside_face]

    # Apply global brightness knob (your request: 0.5 => 50% darker)
    rgb = np.clip(rgb * float(BRIGHTNESS), 0.0, 1.0)

    alpha = np.ones((V.shape[0], 1), dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=1)
    return rgba


# --------------------------
# 5) Render (flat, no shading influence)
# --------------------------
def render_mesh(V, F, vertex_rgba):
    tri_mesh = trimesh.Trimesh(V, F, process=False)
    tri_mesh.visual.vertex_colors = (np.clip(vertex_rgba, 0, 1) * 255).astype(np.uint8)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

    vmin = V.min(axis=0); vmax = V.max(axis=0)
    radius = 0.5 * np.linalg.norm(vmax - vmin) + 1e-6
    yfov = np.deg2rad(YFOV_DEG)
    dist = (radius / np.tan(yfov / 2.0)) * 1.35

    cam_pose = look_at((0.0, 0.0, dist))
    camera = pyrender.PerspectiveCamera(yfov=yfov)

    scene = pyrender.Scene(
        bg_color=[1, 1, 1, 1],
        ambient_light=[1.0, 1.0, 1.0],
    )
    scene.add(mesh)
    scene.add(camera, pose=cam_pose)

    flags = {
        pyrender.RenderFlags.FLAT: True,
        pyrender.RenderFlags.SKIP_CULL_FACES: True,
    }

    pyrender.Viewer(
        scene,
        use_raymond_lighting=False,
        viewport_size=(VIEWPORT_W, VIEWPORT_H),
        camera=camera,
        camera_pose=cam_pose,
        render_flags=flags,
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

    # 1) photo landmarks + face mask
    lm2d, face_mask, oval_pts = detect_landmarks_68_and_facemask(img_bgr)

    # Debug landmarks + face oval
    dbg = img_bgr.copy()
    for (x, y) in lm2d:
        cv2.circle(dbg, (int(x), int(y)), 2, (0, 255, 0), -1)
    poly = np.round(oval_pts).astype(np.int32)
    cv2.polylines(dbg, [poly], isClosed=True, color=(255, 0, 0), thickness=2)  # blue-ish in BGR
    cv2.imwrite(OUT_DEBUG_LMK, dbg)
    print("[OK] wrote", OUT_DEBUG_LMK)

    # Estimate skin fill from face region (so non-mapped area isn't gray)
    skin_rgb01 = estimate_skin_rgb01(img_bgr, face_mask, lo_pct=15.0, hi_pct=85.0)
    print("skin_rgb01:", skin_rgb01)

    # 2) FLAME neutral mesh + 3D landmarks
    V0, F, L3 = flame_neutral_mesh_and_landmarks(device)

    # 3) same centering/rotation to vertices and landmarks
    center = 0.5 * (V0.min(axis=0) + V0.max(axis=0))
    V = V0 - center[None, :]
    L = L3 - center[None, :]

    R_fix = (rot_z(ROT_Z_DEG) @ rot_y(ROT_Y_DEG) @ rot_x(ROT_X_DEG)).astype(np.float32)
    V = (R_fix @ V.T).T
    L = (R_fix @ L.T).T

    # 4) Solve 3D->2D projection
    A, t = solve_affine_3d_to_2d(L, lm2d)

    # Debug overlay: projected FLAME landmarks
    proj = (A @ L.T).T + t[None, :]
    vis = img_bgr.copy()
    for (x, y) in lm2d:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    for (x, y) in proj:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imwrite(OUT_DEBUG_OVERLAY, vis)
    print("[OK] wrote", OUT_DEBUG_OVERLAY)

    # 5) Paint colors using face mask gating
    rgba = sample_vertex_colors(img_bgr, face_mask, V, F, A, t, skin_rgb01)

    # 6) Render
    render_mesh(V, F, rgba)


if __name__ == "__main__":
    main()