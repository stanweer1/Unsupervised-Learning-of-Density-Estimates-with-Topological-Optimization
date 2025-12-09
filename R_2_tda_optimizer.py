import torch
import torch.nn.functional as F
from torch_topological.nn import CubicalComplex, SummaryStatisticLoss
from tqdm import tqdm
import numpy as np

# Compatibility wrapper for persistence info
class PersistenceInfo:
    def __init__(self, diagram):
        from torch_topological.nn.data import PersistenceInformation
        if isinstance(diagram, PersistenceInformation):
            diagram = diagram.diagram

        if diagram is None:
            self.diagram = torch.empty((0, 2), dtype=torch.float32)
            return

        if isinstance(diagram, torch.Tensor):
            if diagram.ndim == 1:
                self.diagram = diagram.view(-1, 2)
            elif diagram.ndim == 2 and diagram.shape[1] == 2:
                self.diagram = diagram
            else:
                self.diagram = diagram.reshape(-1, 2)
            return

        if isinstance(diagram, (int, float)):
            self.diagram = torch.tensor([[diagram, diagram]], dtype=torch.float32)
            return

        if isinstance(diagram, list):
            tensor_list = []
            for d in diagram:
                pi = PersistenceInfo(d)
                if pi.diagram.numel() > 0:
                    tensor_list.append(pi.diagram)
            if tensor_list:
                self.diagram = torch.cat(tensor_list, dim=0)
            else:
                self.diagram = torch.empty((0, 2), dtype=torch.float32)
            return

        raise TypeError(f"Unsupported diagram type: {type(diagram)}")

def kde_density_torch_nd(points, grid_points, bandwidth):
    N_samples, N_dims = points.shape
    N_grid = grid_points.shape[0]
    
    # Move to device of points
    grid_points = grid_points.to(points.device)
    bandwidth = bandwidth.to(points.device)

    points_exp = points.unsqueeze(1)      # (N_samples, 1, N_dims)
    grid_exp = grid_points.unsqueeze(0)     # (1, N_grid, N_dims)
    
    sq_dist = ((points_exp - grid_exp) ** 2).sum(dim=2)  # (N_samples, N_grid)
    
    norm_factor = (2 * torch.pi) ** (N_dims / 2) * (bandwidth ** N_dims)
    kernel_vals = torch.exp(-sq_dist / (2 * bandwidth**2)) / norm_factor
    
    density = kernel_vals.sum(dim=0) / N_samples  # (N_grid,)
    
    return density

def create_grid_nd(points, resolution_per_dim=32, padding=0.2):
    N_samples, N_dims = points.shape
    
    bounds = []
    for d in range(N_dims):
        dim_min = points[:, d].min().item()
        dim_max = points[:, d].max().item()
        dim_range = dim_max - dim_min
        pad = padding * dim_range
        bounds.append((dim_min - pad, dim_max + pad))
    
    grids_1d = []
    for d in range(N_dims):
        grid_1d = torch.linspace(bounds[d][0], bounds[d][1], resolution_per_dim)
        grids_1d.append(grid_1d)
    
    mesh_grids = torch.meshgrid(*grids_1d, indexing='ij')
    
    grid_points = torch.stack([grid.flatten() for grid in mesh_grids], dim=1)
    
    grid_shape = tuple([resolution_per_dim] * N_dims)
    
    return grid_points, grid_shape, bounds

class SmoothTopologicalLoss(torch.nn.Module):
    def __init__(self, 
                 alpha_pe=1.0, 
                 alpha_tp=1.0,
                 lambda_smooth=1.0,
                 optimal_bw_reference=None):
        super().__init__()
        
        self.alpha_pe = alpha_pe
        self.alpha_tp = alpha_tp
        self.lambda_smooth = lambda_smooth
        self.optimal_bw_reference = optimal_bw_reference
        
        self.loss_fn_pe = SummaryStatisticLoss(summary_statistic='persistent_entropy')
        self.loss_fn_tp = SummaryStatisticLoss(summary_statistic='total_persistence')
        
    def compute_soft_feature_count(self, pi_source, temperature=1.0):
        # Ensure tensor is on the same device as the diagrams
        device = 'cpu'
        if pi_source and pi_source[0].diagram.numel() > 0:
            device = pi_source[0].diagram.device
        total_soft_count = torch.tensor(0.0, device=device)
        
        for pi in pi_source:
            if pi.diagram.shape[0] == 0:
                continue
                
            lifetimes = pi.diagram[:, 1] - pi.diagram[:, 0]
            soft_weights = torch.sigmoid((lifetimes) / temperature)
            total_soft_count = total_soft_count + soft_weights.sum()
        
        return total_soft_count + 1.0  # Add 1 to avoid division by zero
    
    def forward(self, pi_source, bandwidth=None):

        tp_raw = self.loss_fn_tp(pi_source)
        soft_count = self.compute_soft_feature_count(pi_source)
                
        smooth_loss = torch.tensor(0.0, device=bandwidth.device)
        if bandwidth is not None and self.optimal_bw_reference is not None:
            optimal = self.optimal_bw_reference
            smooth_loss = ((bandwidth - optimal) / (optimal)) ** 2
        
        total_loss = self.alpha_pe * soft_count - self.alpha_tp * tp_raw + self.lambda_smooth * smooth_loss
        
        return total_loss


def scotts_rule_bandwidth(points):
    n, d = points.shape
    std = points.std(dim=0)
    sigma = torch.mean(std)
    h = sigma * (n ** (-1.0 / (d + 4)))
    return h.item()

def silvermans_rule_bandwidth(points):
    n, d = points.shape
    std = points.std(dim=0)
    sigma = torch.mean(std)
    h = sigma * ((n * (d + 2) / 4.0) ** (-1.0 / (d + 4)))
    return h.item()

def compute_persistence_nd(density_map):
    ndim = density_map.ndim
    cubical = CubicalComplex(dim=ndim)
    density_input = density_map.unsqueeze(0).unsqueeze(0)
    diagrams = cubical(density_input)
    return diagrams

def optimize_bandwidth_nd(points, 
                          resolution_per_dim=32,
                          init_bw=None, 
                          n_iters=200, 
                          lr=0.01,
                          alpha_pe=1.0,
                          alpha_tp=1.0,
                          lambda_smooth=0.0,
                          only_H0=True,
                          verbose=True):
    
    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32)
    
    N_samples, N_dims = points.shape
    
    if N_dims > 2:
        resolution_per_dim = max(10, int(resolution_per_dim / (N_dims ** 0.5)))

    if init_bw is None:
        init_bw = np.mean([scotts_rule_bandwidth(points), silvermans_rule_bandwidth(points)])
          
    from sklearn.neighbors import NearestNeighbors
    k = max(10, N_samples//10)
    nn = NearestNeighbors(n_neighbors=k).fit(points.cpu().numpy())
    distances, _ = nn.kneighbors()
    ref_bw = np.mean(distances[:, -1])
    init_bw = ref_bw

    bandwidth = torch.nn.Parameter(torch.tensor(init_bw, dtype=torch.float32))
    
    grid_points, grid_shape, bounds = create_grid_nd(points, resolution_per_dim)
    
    opt = torch.optim.SGD([bandwidth], lr=lr)
    loss_fn = SmoothTopologicalLoss(
        alpha_pe=alpha_pe,
        alpha_tp=alpha_tp,
        lambda_smooth=lambda_smooth,
        optimal_bw_reference=init_bw
    )
    
    loss_history = []
    
    iterator = tqdm(range(n_iters), desc="Optimizing") if verbose else range(n_iters)
    
    for i in iterator:
        opt.zero_grad()
        
        density_flat = kde_density_torch_nd(points, grid_points, bandwidth)
        density_map = density_flat.reshape(grid_shape)
        density_map = density_map / (density_map.max() + 1e-8) # Normalize
        
        # Use superlevel-set filtration (negative density)
        diagrams = compute_persistence_nd(-density_map)
        pi_source = [PersistenceInfo(d) for d in diagrams]
        
        if only_H0:
            loss = loss_fn([pi_source[0]], bandwidth)
        else:
            loss = loss_fn(pi_source, bandwidth)

        loss.backward()
        opt.step()
        
        with torch.no_grad():
            bandwidth.data.clamp_(0.01, 5.0)
            
        loss_history.append(loss.item())
        
    # print(loss_history)
        
    final_density = density_map.detach()
    
    return bandwidth.item(), loss_history, final_density, grid_shape, bounds