import numpy as np
import gmsh
import meshio
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import dot
from skfem import ElementTetP2
import pyvista as pv
from onshape_client.client import Client
import os
import time
import json
import imageio
import argparse # [추가] CLI 인자 처리를 위한 모듈

# ---------------------------------------------------------
# 0. 폴더 경로 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MESH_DIR = os.path.join(BASE_DIR, "mesh")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for d in [MODEL_DIR, MESH_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------
# 1. 설정 변수
# ---------------------------------------------------------
E_MODULUS = 193e9   
NU_POISSON = 0.29   
FORCE_TOTAL = -30.0 
LOAD_RADIUS = 2.0   

# Onshape 설정
USE_ONSHAPE = True 
API_KEY = "on_Z7RYJe3KS6AC9oa7sjxDg"
SECRET_KEY = "KPUmuM0m96K8GszDShlJ2kjGAVSbdpEEtHH28DZYrXgrw5tt"
DOC_ID = "df807161c9ee767c8a1ef426"
WSP_ID = "37d64a89af67b957f92edb40"
ELM_ID = "7acbe77c1caf5bda3c0ac151" 

# ---------------------------------------------------------
# 2. Onshape Export & Mesh Generation
# ---------------------------------------------------------
def generate_mesh(t_mm):
    step_filename = os.path.join(MODEL_DIR, f"beam_t{t_mm}.step")
    
    if USE_ONSHAPE:
        client = Client(configuration={"base_url": "https://cad.onshape.com", "access_key": API_KEY, "secret_key": SECRET_KEY})
        config = f"Thickness={5.0 + t_mm}+mm"
        print(f"[Onshape] Requesting Translation (STEP) with configuration: {config} ...")
        
        try:
            url_trans = f"/api/partstudios/d/{DOC_ID}/w/{WSP_ID}/e/{ELM_ID}/translations"
            payload = { "formatName": "STEP", "storeInDocument": False, "configuration": config }
            response = client.api_client.call_api(url_trans, 'POST', query_params=[], body=payload, _preload_content=False, response_type=None)
            
            http_resp = response[0] if isinstance(response, tuple) else response
            raw_data = http_resp.data if hasattr(http_resp, 'data') else http_resp.read()
            resp_data = json.loads(raw_data)
            tid = resp_data.get('id')

            print(f"[Onshape] Job started. Translation ID: {tid}")
            url_status = f"/api/translations/{tid}"
            external_data_id = None
            
            for _ in range(20): 
                time.sleep(2) 
                status_resp = client.api_client.call_api(url_status, 'GET', query_params=[], _preload_content=False, response_type=None)
                s_http_resp = status_resp[0] if isinstance(status_resp, tuple) else status_resp
                s_raw_data = s_http_resp.data if hasattr(s_http_resp, 'data') else s_http_resp.read()
                s_data = json.loads(s_raw_data)
                
                state = s_data.get('requestState')
                if state == 'DONE':
                    external_data_id = s_data.get('resultExternalDataIds')[0]
                    break
                elif state == 'FAILED':
                    raise Exception(f"Translation failed.")

            url_download = f"/api/documents/d/{DOC_ID}/externaldata/{external_data_id}" 
            dl_resp = client.api_client.call_api(url_download, 'GET', query_params=[], _preload_content=False, response_type='file')
            data_obj = dl_resp[0] if isinstance(dl_resp, tuple) else dl_resp
            file_data = data_obj.read() if hasattr(data_obj, 'read') else (data_obj.data if hasattr(data_obj, 'data') else data_obj)

            with open(step_filename, "wb") as f:
                f.write(file_data)
            print(f"[Onshape] Saved {step_filename}")

        except Exception as e:
            print(f"[Onshape] Export Failed: {e}")
            raise e 

    gmsh.initialize()
    gmsh.model.add("Beam")
    gmsh.model.occ.importShapes(step_filename)
    gmsh.model.occ.synchronize()

    if USE_ONSHAPE:
        bbox = gmsh.model.getBoundingBox(-1, -1)
        max_dim = max(abs(bbox[3] - bbox[0]), abs(bbox[4] - bbox[1]), abs(bbox[5] - bbox[2]))
        if max_dim > 1.0: 
            print(f"[Gmsh] Scaling to Meters (x0.001)...")
            vols = gmsh.model.getEntities(3)
            gmsh.model.occ.dilate(vols, 0, 0, 0, 0.001, 0.001, 0.001)
            gmsh.model.occ.synchronize()

    current_thickness_mm = 5.0 + t_mm
    target_mesh_size_mm = max(current_thickness_mm / 1.5, 1.0)
    target_mesh_size_m = target_mesh_size_mm * 1e-3
    
    print(f"[Gmsh] Target Mesh Size: {target_mesh_size_mm:.3f} mm")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_mesh_size_m * 0.8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_mesh_size_m * 1.2)
    
    gmsh.model.mesh.generate(3)
    
    msh_filename = os.path.join(MESH_DIR, f"beam_t{t_mm}.msh")
    gmsh.write(msh_filename)
    gmsh.finalize()
    return msh_filename

# ---------------------------------------------------------
# 3. Simulation (scikit-fem)
# ---------------------------------------------------------
def run_simulation(mesh_file, t_mm):
    mesh = Mesh.load(mesh_file)
    print(f"[Simulation] Mesh loaded. Nodes: {mesh.nvertices}")
    
    x_min = mesh.p[0].min()
    y_min, y_max = mesh.p[1].min(), mesh.p[1].max()
    z_max = mesh.p[2].max()

    e = ElementVector(ElementTetP2())
    basis = Basis(mesh, e) 

    lam, mu = lame_parameters(E_MODULUS, NU_POISSON)
    K = asm(linear_elasticity(lam, mu), basis)

    fixed_dofs = basis.get_dofs(lambda x: np.isclose(x[0], x_min, atol=1e-4))
    
    target_x = x_min + 0.045 
    target_y = (y_min + y_max) / 2.0
    target_z = z_max
    
    nodes_x = mesh.p[0, :]
    nodes_y = mesh.p[1, :]
    nodes_z = mesh.p[2, :]
    dist = np.sqrt((nodes_x - target_x)**2 + (nodes_y - target_y)**2 + (nodes_z - target_z)**2)
    
    load_node_indices = np.where(dist < (LOAD_RADIUS * 1e-3))[0]
    if len(load_node_indices) == 0:
        nearest_idx = np.argmin(dist)
        load_node_indices = [nearest_idx]

    f = np.zeros(basis.N)
    force_per_node = FORCE_TOTAL / len(load_node_indices)
    
    for node_idx in load_node_indices:
        if node_idx < len(basis.nodal_dofs[2]):
             dof_z = basis.nodal_dofs[2][node_idx] 
             f[dof_z] += force_per_node

    print("[Simulation] Solving linear system...")
    u = solve(*condense(K, f, D=fixed_dofs))
    print("[Simulation] Solved.")
    
    return mesh, u, basis

# ---------------------------------------------------------
# 4. Dashboard Visualization (Improved)
# ---------------------------------------------------------
def visualize_dashboard(mesh, u, basis, t_mm, fixed_scale=None):
    print("[Vis] Generating Dashboard View...")
    
    # --- Data Interpolation ---
    M = basis.probes(mesh.p)
    u_at_nodes = M @ u
    
    n_vertices = mesh.p.shape[1]
    u_x = u_at_nodes[0 * n_vertices : 1 * n_vertices]
    u_y = u_at_nodes[1 * n_vertices : 2 * n_vertices]
    u_z = u_at_nodes[2 * n_vertices : 3 * n_vertices]
    
    displacement = np.vstack((u_x, u_y, u_z)).T
    
    # --- Create Grid ---
    cells = np.hstack([np.full((mesh.t.shape[1], 1), 4), mesh.t.T]).flatten()
    cell_type = np.full(mesh.t.shape[1], pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_type, mesh.p.T)
    
    grid.point_data["Displacement"] = displacement
    grid.point_data["Magnitude"] = np.linalg.norm(displacement, axis=1)
    
    # --- Scale Factor Calculation ---
    max_disp_real = np.max(grid.point_data['Magnitude'])
    max_disp_mm = max_disp_real * 1000
    
    if fixed_scale is not None:
        # [Fixed Scale] 사용자가 지정한 값 사용
        scale_factor = fixed_scale
        scale_mode_text = f"Fixed (x{scale_factor})"
    else:
        # [Auto Scale] 모델 크기의 15%로 보이게 자동 조절
        bounds = grid.bounds
        model_size = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
        if max_disp_real > 1e-12:
            scale_factor = (model_size * 0.15) / max_disp_real
        else:
            scale_factor = 1.0
        scale_mode_text = f"Auto (x{scale_factor:.1f})"

    print(f"[Vis] Scale Mode: {scale_mode_text}")

    # Warp Mesh
    warped = grid.warp_by_vector("Displacement", factor=scale_factor)

    # --- Dashboard Layout ---
    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600), off_screen=True)
    
    # [수정] 메인 타이틀을 겹치지 않게 별도 처리하거나, 텍스트 위치를 조정
    # 여기서는 subplot 별로 add_title을 사용하여 겹침을 방지함

    # View 1: CAD Geometry
    plotter.subplot(0, 0)
    plotter.add_title(f"1. CAD Geometry (t={5+t_mm:.1f}mm)", font_size=10)
    plotter.add_mesh(grid, color="white", show_edges=False, smooth_shading=True, specular=0.5)
    plotter.view_isometric()

    # View 2: Mesh Structure
    plotter.subplot(0, 1)
    plotter.add_title(f"2. Mesh (Size ~{max((5+t_mm)/1.5, 1.0):.1f}mm)", font_size=10)
    plotter.add_mesh(grid, style='wireframe', color='lightblue', line_width=1.5)
    plotter.view_isometric()

    # View 3: FEM Result
    plotter.subplot(0, 2)
    plotter.add_title(f"3. Result (Max Def.: {max_disp_mm:.4f}mm)", font_size=10)
    plotter.add_mesh(warped, scalars="Magnitude", cmap="jet", show_edges=False)
    
    # 스케일 정보는 우측 하단에 작게 표시
    plotter.add_text(f"Scale: {scale_mode_text}", position='lower_right', font_size=8, color='black')
    plotter.view_isometric()

    plotter.link_views()

    img_filename = os.path.join(OUTPUT_DIR, f"dashboard_t_{t_mm:.1f}.png")
    plotter.screenshot(img_filename)
    print(f"[Vis] Dashboard saved: {img_filename}")
    
    return img_filename

# ---------------------------------------------------------
# Main Loop (With Argument Parser)
# ---------------------------------------------------------
if __name__ == "__main__":
    # [추가] CLI 인자 파싱
    parser = argparse.ArgumentParser(description="FEM Simulation Pipeline")
    parser.add_argument("--scale", type=float, default=None, 
                        help="Set a fixed deformation scale factor (e.g., 200). If not set, uses Auto-Scaling.")
    args = parser.parse_args()

    # Fixed Scale 사용 시 안내 메시지
    if args.scale:
        print(f"\n[System] Using FIXED scale factor: x{args.scale}")
        print("Note: This helps to compare relative stiffness. Thick beams may appear rigid.")
    else:
        print("\n[System] Using AUTO scale factor (Variable Magnification).")
        print("Note: Each result is normalized to look deformed. Good for checking mode shapes.")

    thickness_variations = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] 
    image_files = []

    for t in thickness_variations:
        print(f"\n=== Processing Pipeline for t = {t} ===")
        msh_file = generate_mesh(t)
        mesh_obj, u_sol, basis_obj = run_simulation(msh_file, t)
        
        if u_sol is not None:
            # args.scale을 전달 (None이면 Auto, 값이 있으면 Fixed)
            img_path = visualize_dashboard(mesh_obj, u_sol, basis_obj, t, fixed_scale=args.scale)
            image_files.append(img_path)

    if image_files:
        suffix = "fixed" if args.scale else "auto"
        gif_path = os.path.join(OUTPUT_DIR, f"simulation_process_{suffix}.gif")
        print(f"\n[System] Creating GIF animation: {gif_path} ...")
        
        images = [imageio.v2.imread(f) for f in image_files]
        imageio.mimsave(gif_path, images, loop=0, duration=1000) 
        print("[System] GIF creation complete!")
