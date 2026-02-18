"""
FLAME photo-paint + eye-blink (texture-based, stable)
- Geometry fixed (shape=0, expr=0, pose=0) -> avoids flickering boundaries
- MediaPipe FaceMesh:
    * 68 subset for 3D->2D alignment
    * ordered face oval for face mask
    * eye polygons for blink mask
- Best-fit affine map Q ~= A*P + t
- Precompute vertex->image projection and vertex colors ONCE (stable)
- Blink every N seconds by blending eye-region vertex colors toward skin color
  (looks like eyelids closing, even if FLAME expr has no eyelid blendshape)

Keys:
  q : quit
"""

import os
import time
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
IMAGE_PATH = "face2.png"
OUT_DEBUG_LMK = "debug_landmarks.png"
OUT_DEBUG_OVERLAY = "debug_fit_overlay.png"
OUT_DEBUG_EYES = "debug_eye_masks.png"

# User requested:
ROT_X_DEG = 0.0
ROT_Y_DEG = 0.0
ROT_Z_DEG = 0.0

LANDMARK_POINTS_68 = [
    162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
    71,63,105,66,107,336,296,334,293,301,
    168,197,5,4,75,97,2,326,305,
    33,160,158,133,153,144,362,385,387,263,373,380,
    61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
]

FACE_OVAL_ORDERED = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

# MediaPipe FaceMesh eye contours (commonly used)
LEFT_EYE_POLY = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POLY = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Render controls
VIEWPORT_W, VIEWPORT_H = 900, 900
YFOV_DEG = 22.0
BRIGHTNESS = 1.0
SAMPLE_MODE = "nearest"  # "nearest" or "bilinear"

# Blink animation
BLINK_PERIOD_SEC = 2.0
BLINK_CLOSE_SEC  = 0.10
BLINK_HOLD_SEC   = 0.04
BLINK_OPEN_SEC   = 0.12
BLINK_STRENGTH   = 1.0   # 1.0 = fully replace eyes with skin at full blink. Try 0.7~1.2.

# Fallback skin color
NEUTRAL_RGB = np.array([0.78, 0.65, 0.58], dtype=np.float32)


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
# Blink weight function
# --------------------------
def blink_weight(t, period, close_t, hold_t, open_t):
    """
    Returns w in [0,1] repeating every 'period'.
    0=open, 1=closed
    """
    phase = t % period
    if phase < close_t:
        return phase / max(close_t, 1e-6)
    phase -= close_t
    if phase < hold_t:
        return 1.0
    phase -= hold_t
    if phase < open_t:
        return 1.0 - (phase / max(open_t, 1e-6))
    return 0.0


# --------------------------
# 1) MediaPipe landmarks + masks
# --------------------------
def detect_landmarks_and_masks(img_bgr):
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

    # face mask from ordered oval
    oval_pts = pts468[FACE_OVAL_ORDERED].copy()
    face_poly = np.round(oval_pts).astype(np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)

    # eye masks
    left_eye_pts = np.round(pts468[LEFT_EYE_POLY]).astype(np.int32)
    right_eye_pts = np.round(pts468[RIGHT_EYE_POLY]).astype(np.int32)

    eye_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(eye_mask, [left_eye_pts], 255)
    cv2.fillPoly(eye_mask, [right_eye_pts], 255)

    # (optional) slightly expand eye region so it covers eyelids better
    eye_mask = cv2.dilate(eye_mask, np.ones((5, 5), np.uint8), iterations=1)

    return pts68, face_mask, oval_pts, eye_mask, left_eye_pts, right_eye_pts


def estimate_skin_rgb01(img_bgr, face_mask, lo_pct=15.0, hi_pct=85.0):
    m = face_mask > 0
    pix = img_bgr[m]
    if pix.size == 0:
        return NEUTRAL_RGB.copy()

    p = pix.astype(np.float32)
    y = 0.114 * p[:, 0] + 0.587 * p[:, 1] + 0.299 * p[:, 2]
    lo = np.percentile(y, lo_pct)
    hi = np.percentile(y, hi_pct)
    keep = (y >= lo) & (y <= hi)
    p2 = p[keep] if np.any(keep) else p
    med_bgr = np.median(p2, axis=0)
    return (med_bgr[::-1] / 255.0).astype(np.float32)


# --------------------------
# 2) FLAME neutral mesh + 3D landmarks
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
    expr  = torch.zeros((1, 50), dtype=torch.float32, device=device)
    pose  = torch.zeros((1, 6), dtype=torch.float32, device=device)

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

    V0 = verts[0].detach().cpu().numpy().astype(np.float32)
    L3 = lmk3d[0].detach().cpu().numpy().astype(np.float32)
    F = flamelayer.faces
    return V0, F, L3


# --------------------------
# 3) best-fit affine 3D->2D
# --------------------------
def solve_affine_3d_to_2d(P3, Q2):
    X = np.concatenate([P3, np.ones((P3.shape[0], 1), dtype=np.float32)], axis=1)  # (K,4)
    B, *_ = np.linalg.lstsq(X, Q2, rcond=None)  # (4,2)
    B = B.astype(np.float32)
    A = B[:3, :].T
    t = B[3, :].reshape(2)
    return A, t


# --------------------------
# 4) Precompute vertex projection + base colors
# --------------------------
def precompute_vertex_uv_and_colors(img_bgr, face_mask, eye_mask, V, F, A, t, skin_rgb01):
    h, w = img_bgr.shape[:2]

    # project once
    uv = (A @ V.T).T + t[None, :]
    x = uv[:, 0]
    y = uv[:, 1]

    xi = np.round(x).astype(np.int32)
    yi = np.round(y).astype(np.int32)

    inside_img = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    inside_face = np.zeros((V.shape[0],), dtype=bool)
    valid = np.where(inside_img)[0]
    inside_face[valid] = (face_mask[yi[valid], xi[valid]] > 0)

    inside_eye = np.zeros((V.shape[0],), dtype=bool)
    inside_eye[valid] = (eye_mask[yi[valid], xi[valid]] > 0)

    # start with skin color everywhere
    rgb = np.tile(skin_rgb01[None, :], (V.shape[0], 1)).astype(np.float32)

    idx = np.where(inside_face)[0]
    if idx.size > 0:
        if SAMPLE_MODE == "nearest":
            col_bgr = img_bgr[yi[idx], xi[idx]].astype(np.float32)
            rgb[idx] = (col_bgr[:, ::-1] / 255.0)
        else:
            # bilinear
            xf = np.clip(x[idx], 0, w - 2)
            yf = np.clip(y[idx], 0, h - 2)
            x0 = np.floor(xf).astype(np.int32)
            y0 = np.floor(yf).astype(np.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            wa = (x1 - xf) * (y1 - yf)
            wb = (x1 - xf) * (yf - y0)
            wc = (xf - x0) * (y1 - yf)
            wd = (xf - x0) * (yf - y0)

            Ia = img_bgr[y0, x0].astype(np.float32)
            Ib = img_bgr[y1, x0].astype(np.float32)
            Ic = img_bgr[y0, x1].astype(np.float32)
            Id = img_bgr[y1, x1].astype(np.float32)

            col_bgr = (wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id)
            rgb[idx] = (col_bgr[:, ::-1] / 255.0)

    rgb = np.clip(rgb * float(BRIGHTNESS), 0.0, 1.0)
    return rgb, inside_eye


# --------------------------
# 5) Render loop (texture blink)
# --------------------------
def run_blink_loop(V, F, base_rgb, eye_vertices_mask, skin_rgb01):
    r = pyrender.OffscreenRenderer(viewport_width=VIEWPORT_W, viewport_height=VIEWPORT_H)

    vmin = V.min(axis=0); vmax = V.max(axis=0)
    radius = 0.5 * np.linalg.norm(vmax - vmin) + 1e-6
    yfov = np.deg2rad(YFOV_DEG)
    dist = (radius / np.tan(yfov / 2.0)) * 1.35

    cam_pose = look_at((0.0, 0.0, dist))
    camera = pyrender.PerspectiveCamera(yfov=yfov)

    start = time.time()
    while True:
        now = time.time() - start
        w = blink_weight(now, BLINK_PERIOD_SEC, BLINK_CLOSE_SEC, BLINK_HOLD_SEC, BLINK_OPEN_SEC)
        w = float(np.clip(w * BLINK_STRENGTH, 0.0, 1.0))

        rgb = base_rgb.copy()

        # blink effect: blend eye region toward skin color (eyelid look)
        # open: w=0 -> keep original eye pixels
        # closed: w=1 -> replace by skin
        if np.any(eye_vertices_mask) and w > 0.0:
            rgb[eye_vertices_mask] = (1.0 - w) * rgb[eye_vertices_mask] + w * skin_rgb01[None, :]

        alpha = np.ones((V.shape[0], 1), dtype=np.float32)
        rgba = np.concatenate([rgb, alpha], axis=1)

        tri_mesh = trimesh.Trimesh(V, F, process=False)
        tri_mesh.visual.vertex_colors = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

        scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[1.0, 1.0, 1.0])
        scene.add(mesh)
        scene.add(camera, pose=cam_pose)

        color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)

        cv2.imshow("FLAME blink (texture) - press q", color[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    r.delete()
    cv2.destroyAllWindows()


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
    print("Image:", IMAGE_PATH, "shape:", img_bgr.shape)

    # 1) landmarks + masks
    lm2d, face_mask, oval_pts, eye_mask, left_eye_pts, right_eye_pts = detect_landmarks_and_masks(img_bgr)

    # debug draw
    dbg = img_bgr.copy()
    for (x, y) in lm2d:
        cv2.circle(dbg, (int(x), int(y)), 2, (0, 255, 0), -1)
    poly = np.round(oval_pts).astype(np.int32)
    cv2.polylines(dbg, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(dbg, [left_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
    cv2.polylines(dbg, [right_eye_pts], isClosed=True, color=(0, 255, 255), thickness=1)
    cv2.imwrite(OUT_DEBUG_LMK, dbg)
    print("[OK] wrote", OUT_DEBUG_LMK)

    dbg_eye = cv2.cvtColor(eye_mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(OUT_DEBUG_EYES, dbg_eye)
    print("[OK] wrote", OUT_DEBUG_EYES)

    # 2) skin fill color
    skin_rgb01 = estimate_skin_rgb01(img_bgr, face_mask, lo_pct=15.0, hi_pct=85.0)
    print("skin_rgb01:", skin_rgb01)

    # 3) FLAME mesh
    V0, F, L3 = flame_neutral_mesh_and_landmarks(device)

    # 4) center + rotate (same for V and landmarks)
    center = 0.5 * (V0.min(axis=0) + V0.max(axis=0))
    R_fix = (rot_z(ROT_Z_DEG) @ rot_y(ROT_Y_DEG) @ rot_x(ROT_X_DEG)).astype(np.float32)

    V = (R_fix @ (V0 - center[None, :]).T).T
    L = (R_fix @ (L3 - center[None, :]).T).T

    # 5) affine map
    A, t = solve_affine_3d_to_2d(L, lm2d)

    proj = (A @ L.T).T + t[None, :]
    vis = img_bgr.copy()
    for (x, y) in lm2d:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    for (x, y) in proj:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imwrite(OUT_DEBUG_OVERLAY, vis)
    print("[OK] wrote", OUT_DEBUG_OVERLAY)

    # 6) precompute stable vertex colors + eye vertex mask
    base_rgb, eye_vertices_mask = precompute_vertex_uv_and_colors(
        img_bgr, face_mask, eye_mask, V, F, A, t, skin_rgb01
    )
    print("eye vertices:", int(eye_vertices_mask.sum()), "/", int(V.shape[0]))

    # 7) blink loop (texture-based)
    run_blink_loop(V, F, base_rgb, eye_vertices_mask, skin_rgb01)


if __name__ == "__main__":
    main()