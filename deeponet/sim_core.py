import numpy as np
import gmsh
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem import ElementTetP2
from onshape_client.client import Client
import os
import time
import json
from scipy.interpolate import griddata  # optional (legacy interpolation)

# ---------------------------------------------------------
# Paths / Dirs
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MESH_DIR = os.path.join(BASE_DIR, "mesh")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [MODEL_DIR, MESH_DIR, DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------
# Material / Load settings
# ---------------------------------------------------------
E_MODULUS = 193e9
NU_POISSON = 0.29

FORCE_TOTAL = -50.0   # N  (negative = "down" along thickness axis)
LOAD_RADIUS = 2.0     # mm (concentric load patch radius)
LOAD_RATIO = 0.90     # load location along length (0~1)
FIXED_ATOL = 1e-4     # m tolerance for fixed boundary selection

# Design parameter (thickness offset) range used in dataset/training
T_MIN, T_MAX = -4.0, 3.0  # mm offset (Thickness = 5 + t)

# ---------------------------------------------------------
# Onshape settings (allow env override)
# ---------------------------------------------------------
API_KEY = os.getenv("ONSHAPE_API_KEY", "on_Z7RYJe3KS6AC9oa7sjxDg")
SECRET_KEY = os.getenv("ONSHAPE_SECRET_KEY", "KPUmuM0m96K8GszDShlJ2kjGAVSbdpEEtHH28DZYrXgrw5tt")
DOC_ID = os.getenv("ONSHAPE_DOC_ID", "df807161c9ee767c8a1ef426")
WSP_ID = os.getenv("ONSHAPE_WSP_ID", "37d64a89af67b957f92edb40")
ELM_ID = os.getenv("ONSHAPE_ELM_ID", "7acbe77c1caf5bda3c0ac151")

# ---------------------------------------------------------
# Geometry / coordinate helpers
# ---------------------------------------------------------
def infer_axes_from_points(points_xyz: np.ndarray):
    """
    Infer (length, width, thickness) axis indices from bbox extents.
    points_xyz: (N,3)
    returns: axes = (aL, aW, aT), mins(3,), maxs(3,), spans(3,)
    """
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    spans = maxs - mins
    order = np.argsort(spans)[::-1]  # descending by extent
    aL, aW, aT = int(order[0]), int(order[1]), int(order[2])
    return (aL, aW, aT), mins, maxs, spans


def canonicalize_points(points_xyz: np.ndarray, axes: tuple[int, int, int]) -> np.ndarray:
    """
    Reorder xyz points into canonical (L, W, T) order.
    axes = (aL, aW, aT) where each is one of {0,1,2}.
    """
    aL, aW, aT = axes
    return points_xyz[:, [aL, aW, aT]]


def uncanonicalize_points(points_lwt: np.ndarray, axes: tuple[int, int, int]) -> np.ndarray:
    """
    Inverse of canonicalize_points.
    points_lwt: (N,3) in (L, W, T)
    """
    aL, aW, aT = axes
    out = np.zeros_like(points_lwt)
    out[:, aL] = points_lwt[:, 0]
    out[:, aW] = points_lwt[:, 1]
    out[:, aT] = points_lwt[:, 2]
    return out


def normalize_points(points: np.ndarray, mins: np.ndarray, spans: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """(points - mins) / spans -> [0,1] approximately."""
    return (points - mins) / (spans + eps)


def denormalize_points(points_norm: np.ndarray, mins: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """points_norm * spans + mins"""
    return points_norm * spans + mins


def t_to_branch_input(t_mm: np.ndarray) -> np.ndarray:
    """
    Map thickness offset t (mm) to [-1,1] for branch net stability.
    Accepts scalar or array.
    """
    t_mm = np.asarray(t_mm, dtype=np.float32)
    return 2.0 * (t_mm - T_MIN) / (T_MAX - T_MIN) - 1.0


def branch_input_to_t(t_scaled: np.ndarray) -> np.ndarray:
    """Inverse map of t_to_branch_input."""
    t_scaled = np.asarray(t_scaled, dtype=np.float32)
    return (0.5 * (t_scaled + 1.0) * (T_MAX - T_MIN)) + T_MIN


# ---------------------------------------------------------
# Gmsh / mesh generation
# ---------------------------------------------------------
def _gmsh_init_once():
    if not gmsh.is_initialized():
        gmsh.initialize()


def generate_mesh(t_mm: float, use_onshape: bool = True):
    """
    Download STEP from Onshape (optional), import into Gmsh, create tetra mesh.
    Returns gmsh module instance (in-memory model).
    """
    step_filename = os.path.join(MODEL_DIR, f"beam_t{t_mm:.2f}.step")

    if use_onshape and not os.path.exists(step_filename):
        client = Client(configuration={"base_url": "https://cad.onshape.com",
                                       "access_key": API_KEY,
                                       "secret_key": SECRET_KEY})
        config = f"Thickness={5.0 + t_mm}+mm"

        try:
            url_trans = f"/api/partstudios/d/{DOC_ID}/w/{WSP_ID}/e/{ELM_ID}/translations"
            payload = {"formatName": "STEP", "storeInDocument": False, "configuration": config}
            response = client.api_client.call_api(url_trans, 'POST', query_params=[],
                                                  body=payload, _preload_content=False,
                                                  response_type=None)

            http_resp = response[0] if isinstance(response, tuple) else response
            tid = json.loads(http_resp.data if hasattr(http_resp, 'data') else http_resp.read()).get('id')

            url_status = f"/api/translations/{tid}"
            external_data_id = None

            for _ in range(30):
                time.sleep(1.0)
                status_resp = client.api_client.call_api(url_status, 'GET', query_params=[],
                                                         _preload_content=False, response_type=None)
                s_data = json.loads(status_resp[0].data if hasattr(status_resp[0], 'data') else status_resp[0].read())

                if s_data.get('requestState') == 'DONE':
                    external_data_id = s_data.get('resultExternalDataIds')[0]
                    break
                elif s_data.get('requestState') == 'FAILED':
                    raise RuntimeError("Onshape translation failed.")

            if external_data_id is None:
                raise RuntimeError("Onshape translation timeout.")

            url_dl = f"/api/documents/d/{DOC_ID}/externaldata/{external_data_id}"
            dl_resp = client.api_client.call_api(url_dl, 'GET', query_params=[],
                                                 _preload_content=False, response_type='file')
            data_obj = dl_resp[0]
            file_data = data_obj.read() if hasattr(data_obj, 'read') else data_obj.data

            with open(step_filename, "wb") as f:
                f.write(file_data)

        except Exception as e:
            print(f"[Onshape] Failed: {e}")
            return None

    _gmsh_init_once()

    gmsh.clear()
    gmsh.model.add("Beam")
    gmsh.model.occ.importShapes(step_filename)
    gmsh.model.occ.synchronize()

    # Unit scaling to meters if geometry is in mm (or generally "too big")
    bbox = gmsh.model.getBoundingBox(-1, -1)  # (xmin, ymin, zmin, xmax, ymax, zmax)
    spans = np.array([abs(bbox[3] - bbox[0]), abs(bbox[4] - bbox[1]), abs(bbox[5] - bbox[2])], dtype=float)
    if spans.max() > 1.0:
        gmsh.model.occ.dilate(gmsh.model.getEntities(3), 0, 0, 0, 0.001, 0.001, 0.001)
        gmsh.model.occ.synchronize()

    # Mesh size based on thickness (mm -> m)
    current_thickness_mm = 5.0 + t_mm
    size_mm = max(current_thickness_mm / 1.5, 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_mm * 1e-3 * 0.8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_mm * 1e-3 * 1.2)
    gmsh.option.setNumber("General.Verbosity", 1)

    gmsh.model.mesh.generate(3)
    return gmsh


def load_mesh_from_gmsh(gmsh_instance, msh_filename: str = "temp_current.msh") -> Mesh:
    """Write current gmsh model to .msh and load as skfem Mesh."""
    temp_msh = os.path.join(MESH_DIR, msh_filename)
    gmsh_instance.write(temp_msh)
    return Mesh.load(temp_msh)


# ---------------------------------------------------------
# FEM solve
# ---------------------------------------------------------
def run_simulation(gmsh_instance):
    """
    Convert gmsh -> skfem mesh, solve linear elasticity with:
      - clamped at fixed end (min along length axis)
      - concentric load patch at LOAD_RATIO*L on top surface, directed "down" along thickness axis
    Returns: mesh, basis, u, meta(dict)
    """
    mesh = load_mesh_from_gmsh(gmsh_instance)

    # Vector P2 element
    e = ElementVector(ElementTetP2())
    basis = Basis(mesh, e)

    # Infer axes from mesh vertices
    points_xyz = mesh.p.T  # (Nverts, 3)
    axes, mins_xyz, maxs_xyz, spans_xyz = infer_axes_from_points(points_xyz)
    aL, aW, aT = axes

    # Assemble stiffness
    lam, mu = lame_parameters(E_MODULUS, NU_POISSON)
    K = asm(linear_elasticity(lam, mu), basis)

    # Fixed end: coordinate along length axis equals min
    L_min = mesh.p[aL].min()
    fixed_dofs = basis.get_dofs(lambda x: np.isclose(x[aL], L_min, atol=FIXED_ATOL))

    # Load target position
    L_max = mesh.p[aL].max()
    L = L_max - L_min
    load_pos_L = L_min + LOAD_RATIO * L

    W_center = 0.5 * (mesh.p[aW].min() + mesh.p[aW].max())
    T_top = mesh.p[aT].max()  # top surface

    target = np.zeros(3, dtype=float)
    target[aL] = load_pos_L
    target[aW] = W_center
    target[aT] = T_top

    # Select nodes within radius (meters)
    nodes_pos = points_xyz
    dist = np.linalg.norm(nodes_pos - target[None, :], axis=1)
    load_indices = np.where(dist < (LOAD_RADIUS * 1e-3))[0]
    if load_indices.size == 0:
        load_indices = np.array([int(np.argmin(dist))], dtype=int)

    # Build force vector
    f = np.zeros(basis.N, dtype=float)
    f_val = FORCE_TOTAL / float(load_indices.size)

    force_component = aT  # apply along thickness axis
    for idx in load_indices:
        # basis.nodal_dofs[comp][i] -> dof index for component comp at i-th vertex
        if idx < len(basis.nodal_dofs[force_component]):
            f[basis.nodal_dofs[force_component][idx]] += f_val

    # Solve
    u = solve(*condense(K, f, D=fixed_dofs))

    meta = {
        "axes": axes,
        "mins_xyz": mins_xyz,
        "maxs_xyz": maxs_xyz,
        "spans_xyz": spans_xyz,
        "length_axis": aL,
        "width_axis": aW,
        "thickness_axis": aT,
        "L_min": float(L_min),
        "L_max": float(L_max),
        "load_pos_L": float(load_pos_L),
        "force_component": int(force_component),
        "n_load_nodes": int(load_indices.size),
    }
    return mesh, basis, u, meta


# ---------------------------------------------------------
# Legacy interpolation (kept for backward compatibility)
# ---------------------------------------------------------
def interpolate_to_points(mesh, basis, u, query_points):
    """
    SciPy griddata interpolation:
      - query_points: (3, N) in xyz meters
      - returns (N,3) displacement at those points
    Outside mesh -> fill 0.
    """
    try:
        M = basis.probes(mesh.p)
        u_at_nodes = M @ u  # (3*Nverts,)
        n_verts = mesh.p.shape[1]
        u_x = u_at_nodes[0:n_verts]
        u_y = u_at_nodes[n_verts:2*n_verts]
        u_z = u_at_nodes[2*n_verts:]
        src_values = np.stack([u_x, u_y, u_z], axis=1)
        src_points = mesh.p.T

        u_interp = griddata(
            points=src_points,
            values=src_values,
            xi=query_points.T,
            method='linear',
            fill_value=0.0
        )
        return u_interp
    except Exception as e:
        print(f"[Interpolation Warning] SciPy failed: {e}")
        return np.zeros((query_points.shape[1], 3), dtype=float)


# ---------------------------------------------------------
# Reference grid (canonical / normalized)
# ---------------------------------------------------------
def get_reference_grid(nx: int = 50, ny: int = 10, nz: int = 10, to_minus1_1: bool = True) -> np.ndarray:
    """
    Return a fixed grid in canonical normalized coordinates (L,W,T).
      - If to_minus1_1=True -> values in [-1,1]
      - else -> values in [0,1]
    This grid is geometry-independent and safe to use for debugging,
    but for accurate inference on a specific geometry, prefer mesh vertices.
    """
    s = np.linspace(0.0, 1.0, nx)
    w = np.linspace(0.0, 1.0, ny)
    t = np.linspace(0.0, 1.0, nz)
    ss, ww, tt = np.meshgrid(s, w, t, indexing="xy")
    pts = np.vstack([ss.ravel(), ww.ravel(), tt.ravel()]).T.astype(np.float32)

    if to_minus1_1:
        pts = 2.0 * pts - 1.0
    return pts
