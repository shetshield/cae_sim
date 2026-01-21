import numpy as np
import gmsh
import meshio
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import dot
from skfem import ElementTetP2
import pyvista as pv
from onshape_client.client import Client
from onshape_client.oas.api.part_studios_api import PartStudiosApi
from onshape_client.oas.models.bt_export_model_params import BTExportModelParams
import os
import time
import json

# ---------------------------------------------------------
# 0. 폴더 경로 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MESH_DIR = os.path.join(BASE_DIR, "mesh")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for d in [MODEL_DIR, MESH_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[System] Directories ready: \n - {MODEL_DIR}\n - {MESH_DIR}\n - {OUTPUT_DIR}")

# ---------------------------------------------------------
# 1. 설정 변수
# ---------------------------------------------------------
E_MODULUS = 193e9   # 193 GPa (SUS304)
NU_POISSON = 0.29   # Poisson's ratio
FORCE_TOTAL = -30.0 # N
LOAD_RADIUS = 2.0   # mm 

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
        client = Client(configuration={"base_url": "https://cad.onshape.com",
                                       "access_key": API_KEY, "secret_key": SECRET_KEY})
        
        config = f"Thickness={5.0 + t_mm}+mm"
        print(f"[Onshape] Requesting Translation (STEP) with configuration: {config} ...")
        
        try:
            url_trans = f"/api/partstudios/d/{DOC_ID}/w/{WSP_ID}/e/{ELM_ID}/translations"
            payload = { "formatName": "STEP", "storeInDocument": False, "configuration": config }
            
            response = client.api_client.call_api(
                url_trans, 'POST', query_params=[], body=payload, _preload_content=False, response_type=None 
            )
            
            http_resp = response[0] if isinstance(response, tuple) else response
            raw_data = http_resp.data if hasattr(http_resp, 'data') else http_resp.read()
            resp_data = json.loads(raw_data)
            tid = resp_data.get('id')

            print(f"[Onshape] Job started. Translation ID: {tid}")

            url_status = f"/api/translations/{tid}"
            external_data_id = None
            
            for _ in range(20): 
                time.sleep(2) 
                status_resp = client.api_client.call_api(
                    url_status, 'GET', query_params=[], _preload_content=False, response_type=None
                )
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
            print(f"[Onshape] Downloading file (ID: {external_data_id})...")
            
            dl_resp = client.api_client.call_api(
                url_download, 'GET', query_params=[], _preload_content=False, response_type='file'
            )
            data_obj = dl_resp[0] if isinstance(dl_resp, tuple) else dl_resp
            file_data = data_obj.read() if hasattr(data_obj, 'read') else (data_obj.data if hasattr(data_obj, 'data') else data_obj)

            with open(step_filename, "wb") as f:
                f.write(file_data)
            print(f"[Onshape] Saved {step_filename} (Size: {os.path.getsize(step_filename)} bytes)")

        except Exception as e:
            print(f"[Onshape] Export Failed: {e}")
            raise e 

    # Meshing
    if USE_ONSHAPE:
        gmsh.initialize()
        gmsh.model.add("Beam")
        gmsh.model.occ.importShapes(step_filename)
        gmsh.model.occ.synchronize()

        bbox = gmsh.model.getBoundingBox(-1, -1)
        max_dim = max(abs(bbox[3] - bbox[0]), abs(bbox[4] - bbox[1]), abs(bbox[5] - bbox[2]))
        if max_dim > 1.0: 
            print(f"[Gmsh] Scaling to Meters (x0.001)...")
            vols = gmsh.model.getEntities(3)
            gmsh.model.occ.dilate(vols, 0, 0, 0, 0.001, 0.001, 0.001)
            gmsh.model.occ.synchronize()

    current_thickness_mm = 5.0 + t_mm
    # P2 요소용 메쉬 사이즈 최적화
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
    
    x_min, x_max = mesh.p[0].min(), mesh.p[0].max()
    y_min, y_max = mesh.p[1].min(), mesh.p[1].max()
    z_min, z_max = mesh.p[2].min(), mesh.p[2].max()

    e = ElementVector(ElementTetP2())
    basis = Basis(mesh, e) # Basis 생성

    lam, mu = lame_parameters(E_MODULUS, NU_POISSON)
    K = asm(linear_elasticity(lam, mu), basis)

    # 경계 조건
    fixed_dofs = basis.get_dofs(lambda x: np.isclose(x[0], x_min, atol=1e-4))
    
    # 하중 조건
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
        # P2 요소라도 꼭짓점 노드는 존재하므로 해당 인덱스에 하중 적용 가능
        # Basis의 nodal_dofs를 통해 정확한 DOF 인덱스를 찾아야 함
        # nodal_dofs[2]는 Z방향 DOF들
        if node_idx < len(basis.nodal_dofs[2]):
             dof_z = basis.nodal_dofs[2][node_idx] 
             f[dof_z] += force_per_node

    print("[Simulation] Solving linear system...")
    u = solve(*condense(K, f, D=fixed_dofs))
    print("[Simulation] Solved.")
    
    # [중요] basis도 함께 리턴해야 interpolate 가능
    return mesh, u, basis

# ---------------------------------------------------------
# 4. Visualization (PyVista) - Auto-scaling 적용
# ---------------------------------------------------------
def visualize(mesh, u, basis, t_mm):
    print("[Vis] Interpolating solution to mesh vertices...")
    
    # 1. 해(Solution)를 메쉬 꼭짓점으로 보간
    M = basis.probes(mesh.p)
    u_at_nodes = M @ u
    
    n_vertices = mesh.p.shape[1]
    u_x = u_at_nodes[0 * n_vertices : 1 * n_vertices]
    u_y = u_at_nodes[1 * n_vertices : 2 * n_vertices]
    u_z = u_at_nodes[2 * n_vertices : 3 * n_vertices]
    
    displacement = np.vstack((u_x, u_y, u_z)).T
    
    # 2. PyVista Grid 생성
    cells = np.hstack([np.full((mesh.t.shape[1], 1), 4), mesh.t.T]).flatten()
    cell_type = np.full(mesh.t.shape[1], pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_type, mesh.p.T)
    
    # 데이터 매핑
    grid.point_data["Displacement"] = displacement
    grid.point_data["Magnitude"] = np.linalg.norm(displacement, axis=1)
    
    # ---------------------------------------------------------
    # [수정] 자동 스케일링 (Auto-Scaling Factor) 계산
    # ---------------------------------------------------------
    # 목표: 최대 변형량이 빔 전체 길이의 15% 정도로 보이게 과장함
    max_disp_real = np.max(grid.point_data['Magnitude'])
    
    # 빔의 대략적인 크기 (Bounding Box 대각선 길이)
    bounds = grid.bounds
    model_size = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
    
    if max_disp_real > 1e-12: # 변형이 거의 없는 경우 제외
        # (모델크기 * 0.15) / 실제변형량 = 확대배율
        scale_factor = (model_size * 0.15) / max_disp_real
    else:
        scale_factor = 1.0

    print(f"[Vis] Real Max Deflection: {max_disp_real*1000:.4f} mm")
    print(f"[Vis] Applying Deformation Scale Factor: x{scale_factor:.2f}")

    # Warp (과장된 배율 적용)
    warped = grid.warp_by_vector("Displacement", factor=scale_factor) 
    
    # ---------------------------------------------------------
    # 렌더링
    plotter = pv.Plotter(off_screen=True)
    
    # 휘어진 메쉬 추가
    plotter.add_mesh(warped, scalars="Magnitude", cmap="jet", show_edges=False)
    
    # (선택사항) 비교를 위해 변형 전 원본 형상을 흐릿하게(Wireframe) 같이 표시
    # plotter.add_mesh(grid, style='wireframe', color='white', opacity=0.3, label='Original')

    max_disp_mm = max_disp_real * 1000
    info_text = (f"Thickness: {5+t_mm:.2f} mm\n"
                 f"Max Deflection: {max_disp_mm:.4f} mm\n"
                 f"(Vis Scale: x{scale_factor:.1f})")
    
    plotter.add_text(info_text, font_size=10)
    plotter.view_isometric()
    
    img_filename = os.path.join(OUTPUT_DIR, f"result_t_{t_mm:.1f}.png")
    plotter.screenshot(img_filename)
    print(f"[Vis] Saved {img_filename}")

# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    thickness_variations = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
    
    for t in thickness_variations:
        print(f"--- Processing t = {t} ---")
        msh_file = generate_mesh(t)
        # return 값 3개 (mesh, u, basis) 받기
        mesh_obj, u_sol, basis_obj = run_simulation(msh_file, t)
        
        if u_sol is not None:
            # visualize에 basis_obj 전달
            visualize(mesh_obj, u_sol, basis_obj, t)
