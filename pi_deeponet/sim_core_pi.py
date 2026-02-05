from __future__ import annotations

import os
import time
import json
import math
from typing import Optional, Tuple, Dict

import numpy as np
import gmsh
from onshape_client.client import Client
from skfem import Mesh, Basis, asm, condense, solve, ElementVector
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem import ElementTetP2

from pi_config import BeamConfig


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MESH_DIR = os.path.join(BASE_DIR, "mesh")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_pi")

for d in [MODEL_DIR, MESH_DIR, DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------
# Onshape helpers (environment variables recommended)
# ---------------------------------------------------------

def _get_env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v and len(v.strip()) > 0 else None


def get_onshape_config() -> Optional[Dict[str, str]]:
    key = "on_Z7RYJe3KS6AC9oa7sjxDg"
    sec = "KPUmuM0m96K8GszDShlJ2kjGAVSbdpEEtHH28DZYrXgrw5tt"
    doc = "df807161c9ee767c8a1ef426"
    wsp = "37d64a89af67b957f92edb40"
    elm = "7acbe77c1caf5bda3c0ac151"
    if not all([key, sec, doc, wsp, elm]):
        return None
    return {"api_key": key, "secret_key": sec, "doc_id": doc, "wsp_id": wsp, "elm_id": elm}


def download_step_from_onshape(step_path: str, t_mm: float, cfg: BeamConfig) -> bool:
    conf = get_onshape_config()
    if conf is None:
        print("[Onshape] Missing env vars. Set ONSHAPE_* or disable --use_onshape.")
        return False

    client = Client(configuration={
        "base_url": "https://cad.onshape.com",
        "access_key": conf["api_key"],
        "secret_key": conf["secret_key"],
    })

    # Onshape configuration string (depends on your Part Studio setup)
    thickness_mm = cfg.thickness0_mm + t_mm
    config = f"Thickness={thickness_mm}+mm"

    url_trans = f"/api/partstudios/d/{conf['doc_id']}/w/{conf['wsp_id']}/e/{conf['elm_id']}/translations"
    payload = {"formatName": "STEP", "storeInDocument": False, "configuration": config}

    try:
        resp = client.api_client.call_api(url_trans, "POST", query_params=[], body=payload, _preload_content=False, response_type=None)
        http_resp = resp[0] if isinstance(resp, tuple) else resp
        tid = json.loads(http_resp.data if hasattr(http_resp, "data") else http_resp.read()).get("id")

        url_status = f"/api/translations/{tid}"
        external_data_id = None
        for _ in range(60):
            time.sleep(1.0)
            status_resp = client.api_client.call_api(url_status, "GET", query_params=[], _preload_content=False, response_type=None)
            s_data = json.loads(status_resp[0].data if hasattr(status_resp[0], "data") else status_resp[0].read())
            state = s_data.get("requestState")
            if state == "DONE":
                external_data_id = s_data.get("resultExternalDataIds")[0]
                break
            if state == "FAILED":
                raise RuntimeError("Onshape translation FAILED.")

        if external_data_id is None:
            raise RuntimeError("Onshape translation timeout.")

        url_dl = f"/api/documents/d/{conf['doc_id']}/externaldata/{external_data_id}"
        dl_resp = client.api_client.call_api(url_dl, "GET", query_params=[], _preload_content=False, response_type="file")
        data_obj = dl_resp[0]
        file_data = data_obj.read() if hasattr(data_obj, "read") else data_obj.data

        with open(step_path, "wb") as f:
            f.write(file_data)
        return True

    except Exception as e:
        print(f"[Onshape] Failed: {e}")
        return False


# ---------------------------------------------------------
# Mesh / FEM
# ---------------------------------------------------------

def generate_mesh(t_mm: float, cfg: BeamConfig, use_onshape: bool = True) -> Optional[gmsh]:
    """Fetch STEP (if needed) and build gmsh model."""
    step_path = os.path.join(MODEL_DIR, f"beam_t{t_mm:.2f}.step")

    # remove corrupt 0-byte file
    if os.path.exists(step_path) and os.path.getsize(step_path) == 0:
        os.remove(step_path)

    if use_onshape and not os.path.exists(step_path):
        ok = download_step_from_onshape(step_path, t_mm, cfg)
        if not ok:
            return None

    if not os.path.exists(step_path):
        print(f"[Mesh] STEP not found: {step_path}")
        return None

    if not gmsh.is_initialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add("Beam")
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    # scale to meters if seems in mm
    bbox = gmsh.model.getBoundingBox(-1, -1)
    if max(abs(bbox[3] - bbox[0]), abs(bbox[4] - bbox[1]), abs(bbox[5] - bbox[2])) > 1.0:
        gmsh.model.occ.dilate(gmsh.model.getEntities(3), 0, 0, 0, 0.001, 0.001, 0.001)
        gmsh.model.occ.synchronize()

    # mesh size heuristic based on thickness
    thickness_mm = cfg.thickness0_mm + t_mm
    size_mm = max(thickness_mm / 1.5, 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_mm * 1e-3 * 0.8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_mm * 1e-3 * 1.2)
    gmsh.option.setNumber("General.Verbosity", 1)

    gmsh.model.mesh.generate(3)
    return gmsh


def infer_axes(mesh: Mesh) -> Tuple[int, int, int]:
    """Infer (length, width, thickness) axes by bbox span."""
    spans = mesh.p.max(axis=1) - mesh.p.min(axis=1)  # (3,)
    axis_len = int(np.argmax(spans))
    axis_thk = int(np.argmin(spans))
    axis_wid = int([a for a in [0, 1, 2] if a not in [axis_len, axis_thk]][0])
    return axis_len, axis_wid, axis_thk


def run_simulation(gmsh_instance: gmsh, cfg: BeamConfig) -> Tuple[Mesh, Basis, np.ndarray]:
    """Solve linear elasticity with clamped face and concentrated patch load."""
    temp_msh = os.path.join(MESH_DIR, "temp_current.msh")
    gmsh_instance.write(temp_msh)

    mesh = Mesh.load(temp_msh)
    e = ElementVector(ElementTetP2())
    basis = Basis(mesh, e)

    lam, mu = lame_parameters(cfg.E, cfg.nu)
    K = asm(linear_elasticity(lam, mu), basis)

    axis_len, axis_wid, axis_thk = infer_axes(mesh)
    coord = mesh.p  # (3, Nnodes)

    x_min = coord[axis_len].min()
    x_max = coord[axis_len].max()
    L = x_max - x_min
    x_fix = x_min

    # clamp face: axis_len == x_min
    fixed_dofs = basis.get_dofs(lambda x: np.isclose(x[axis_len], x_fix, atol=1e-6))

    # load center at 90% along length, centered in width, on top surface
    y_mid = 0.5 * (coord[axis_wid].min() + coord[axis_wid].max())
    z_top = coord[axis_thk].max()
    x_load = x_min + cfg.load_ratio * L

    target = np.array([0.0, 0.0, 0.0], dtype=float)
    target[axis_len] = x_load
    target[axis_wid] = y_mid
    target[axis_thk] = z_top

    nodes_pos = coord.T  # (N,3)
    dist = np.linalg.norm(nodes_pos - target[None, :], axis=1)
    load_nodes = np.where(dist < cfg.load_radius_m)[0]
    if len(load_nodes) == 0:
        load_nodes = np.array([int(np.argmin(dist))], dtype=int)

    f = np.zeros(basis.N)
    f_val = cfg.force_total_N / len(load_nodes)
    # thickness direction component is axis_thk; but basis.nodal_dofs stores component order [0,1,2] in global axes.
    # We apply force in -thickness global axis direction.
    for idx in load_nodes:
        # nodal_dofs[comp][node_index] exists for vertices; may not for all P2 nodes.
        if idx < len(basis.nodal_dofs[axis_thk]):
            f[basis.nodal_dofs[axis_thk][idx]] += f_val

    u = solve(*condense(K, f, D=fixed_dofs))
    return mesh, basis, u


def extract_vertex_displacement(mesh: Mesh, basis: Basis, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords_vertices, disp_vertices) in global xyz order.

    coords: (N,3), disp: (N,3)
    """
    coords = mesh.p.T  # (N,3) vertices coordinates in global axes
    M = basis.probes(mesh.p)
    u_flat = M @ u
    n = coords.shape[0]
    ux = u_flat[0:n]
    uy = u_flat[n:2*n]
    uz = u_flat[2*n:]
    disp = np.stack([ux, uy, uz], axis=1)
    return coords, disp


# ---------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------

def canonicalize_points_and_vectors(coords: np.ndarray, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize to (Length, Width, Thickness) axes using bbox spans.

    Returns:
        x_hat: (N,3) normalized coords in [-1,1] with canonical axis order
        v_canon: (N,3) vector in canonical component order
        spans: (3,) physical spans in meters (L,W,T)
        center: (3,) bbox center in meters
        perm: (3,) mapping from canonical axes -> global axes indices
              perm = [axis_len, axis_wid, axis_thk]
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans_global = maxs - mins
    axis_len = int(np.argmax(spans_global))
    axis_thk = int(np.argmin(spans_global))
    axis_wid = int([a for a in [0,1,2] if a not in [axis_len, axis_thk]][0])
    perm = np.array([axis_len, axis_wid, axis_thk], dtype=np.int64)

    coords_c = coords[:, perm]  # (N,3) in (L,W,T) order
    vec_c = vec[:, perm]

    mins_c = coords_c.min(axis=0)
    maxs_c = coords_c.max(axis=0)
    spans = maxs_c - mins_c
    spans = np.maximum(spans, 1e-12)
    center = 0.5 * (mins_c + maxs_c)

    x_hat = 2.0 * (coords_c - center) / spans
    x_hat = np.clip(x_hat, -1.0, 1.0)

    return x_hat, vec_c, spans.astype(np.float32), center.astype(np.float32), perm
