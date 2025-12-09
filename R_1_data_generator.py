import numpy as np
from scipy.stats import norm, cauchy

# ===================================================================
# 1D Data
# ===================================================================

def metropolis_hastings_1D(unnormalized_density, n_samples, start_point=0.0, proposal_width=1.0):
    samples = np.zeros((n_samples, 1))
    current_x = start_point
    current_density = unnormalized_density(current_x) + 1e-9 
    
    for i in range(n_samples):
        proposed_x = current_x + proposal_width * np.random.randn()
        proposed_density = unnormalized_density(proposed_x) + 1e-9
        
        acceptance_ratio = proposed_density / current_density
        
        if np.random.rand() < acceptance_ratio:
            current_x = proposed_x
            current_density = proposed_density
            
        samples[i] = current_x
        
    return samples

def generate_samples_1D(unnormalized_density, n_samples=10000, burn_in=1000, start_point=0.0, proposal_width=3.0):
    all_samples = metropolis_hastings_1D(
        unnormalized_density, 
        n_samples + burn_in, 
        start_point, 
        proposal_width
    )
    samples = all_samples[burn_in:]
    return samples

def complex_mixture_pdf(x):
    pdf1 = norm.pdf(x, loc=-4, scale=0.4)
    pdf2 = norm.pdf(x, loc=0, scale=1)
    pdf3 = cauchy.pdf(x, loc=6, scale=0.1)
    return pdf1 + pdf2 + 0.2*pdf3

def bimodal_gaussian_pdf(x):
    pdf1 = norm.pdf(x, loc=-1, scale=0.2)
    pdf2 = norm.pdf(x, loc=1, scale=0.2)
    return pdf1 + pdf2

# ===================================================================
# 2D Data
# ===================================================================

def metropolis_hastings_2D(unnormalized_density, n_samples, proposal_width=1):
    samples = np.zeros((n_samples, 2))
    current_x, current_y = np.random.randn(2)
    current_density = unnormalized_density(current_x, current_y) + 1e-9
    
    for i in range(n_samples):
        proposed_x = current_x + proposal_width * np.random.randn()
        proposed_y = current_y + proposal_width * np.random.randn()
        proposed_density = unnormalized_density(proposed_x, proposed_y) + 1e-9
        
        acceptance_ratio = proposed_density / current_density
        
        if np.random.rand() < acceptance_ratio:
            current_x, current_y = proposed_x, proposed_y
            current_density = proposed_density
            
        samples[i] = [current_x, current_y]
        
    return samples

def generate_samples_2D(unnormalized_density, n_samples=10000, burn_in=1000, proposal_width=0.5):
    all_samples = metropolis_hastings_2D(unnormalized_density, n_samples + burn_in, proposal_width)
    samples = all_samples[burn_in:]
    return samples

def annulus_pdf(x, y):
    r_squared = x**2 + y**2
    return np.exp(-(r_squared**2 - 2 * 2**2 * r_squared))

def clusters_pdf(x, y):
    density = np.zeros_like(x)
    
    for cluster_mean in [[0, 0], [2, 2], [3, -1]]:
        cov = np.array([[0.5, 0], [0, 0.5]])       
        diff = np.stack([x, y], axis=-1) - cluster_mean
        inv_cov = np.linalg.inv(cov)
        exponent = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
        density += np.exp(-0.5 * exponent) / np.sqrt(np.linalg.det(cov))
        
    return density

def elliptical_pdf(X, Y):
    cluster1 = np.exp(-((X+1)**2 / 0.2 + (Y+1)**2 / 5))
    cluster2 = np.exp(-((X-1)**2 / 5 + (Y-1)**2 / 0.1**2))
    return cluster1 + cluster2
    
def weibull_pdf(x, y, alpha=2, beta=4):
    r = np.sqrt(x**2 + y**2)
    return (alpha/beta) * (r/beta)**(alpha - 1) * np.exp(-(r/beta)**alpha)

# ===================================================================
# MNIST Data
# ===================================================================
def scale_points_to_linspaces(points_px, linspaces):
    points_scaled = np.zeros_like(points_px, dtype=np.float64)
    for d in range(points_px.shape[1]):
        L = linspaces[d]
        a, b = L[0], L[-1]
        res = len(L)
        points_scaled[:, d] = a + (points_px[:, d] / (res - 1.0)) * (b - a)
    return points_scaled

def img_to_weighted_points(img):
    h, w = img.shape
    flat = img.flatten().astype(float)
    flat = np.power(flat + 1e-6, 10.0)
    total = flat.sum()
    if total <= 0:
        raise ValueError("Image has zero total intensity; cannot sample.")
    flat /= total
    idx = np.random.choice(h * w, size=100, p=flat)
    ys = idx // w
    xs = idx % w
    points = np.stack([xs, ys], axis=1)
    return points

def load_mnist_digit_resized(digit):
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.GaussianBlur(kernel_size=5, sigma=0.00001),
        transforms.ToTensor()
    ])
    mnist = MNIST(root='./data', train=True, download=True, transform=transform)
    for i in range(len(mnist)):
        img, label = mnist[i]
        if label == digit:
            return img.squeeze(0).numpy()
    raise ValueError("Digit not found.")

def load_mnist_true_density(digit):
    from R_3_kde_metrics import normalize_pdf_nd
    img = load_mnist_digit_resized(digit)
    linspaces = [
        np.linspace(-10, 10, img.shape[0]),
        np.linspace(-10, 10, img.shape[1]),
    ]
    Z_true_norm = normalize_pdf_nd(img, linspaces)
    # Z_true_norm = Z_true_norm.T / Z_true_norm.sum()
    return Z_true_norm, linspaces

def load_mnist_pointcloud(digit):
    img = load_mnist_digit_resized(digit)
    points_px = img_to_weighted_points(img)
    linspaces = [
        np.linspace(-10, 10, img.shape[0]),
        np.linspace(-10, 10, img.shape[1]),
    ]
    data_nd = scale_points_to_linspaces(points_px, linspaces)
    return data_nd

# ===================================================================
# 3D Data
# ===================================================================

def metropolis_hastings_3D(unnormalized_density, n_samples, proposal_width=1):
    samples = np.zeros((n_samples, 3))
    current_x, current_y, current_z = np.random.randn(3)
    current_density = unnormalized_density(current_x, current_y, current_z) + 1e-9
    
    for i in range(n_samples):
        proposed_x = current_x + proposal_width * np.random.randn()
        proposed_y = current_y + proposal_width * np.random.randn()
        proposed_z = current_z + proposal_width * np.random.randn()
        proposed_density = unnormalized_density(proposed_x, proposed_y, proposed_z) + 1e-9
        
        acceptance_ratio = proposed_density / current_density
        
        if np.random.rand() < acceptance_ratio:
            current_x, current_y, current_z = proposed_x, proposed_y, proposed_z
            current_density = proposed_density
            
        samples[i] = [current_x, current_y, current_z]
        
    return samples

def generate_samples_3D(unnormalized_density, n_samples=10000, burn_in=1000, proposal_width=0.5):
    all_samples = metropolis_hastings_3D(unnormalized_density, n_samples + burn_in, proposal_width)
    samples = all_samples[burn_in:]
    return samples

def gauss_pdf_3(x, y, z, h=0):
    return np.exp(-10*((x**2 - h)**2 + (y**2 - h)**2 + (z**2 - h)**2))

def heavytail_spike_3(x, y, z):
    spike = np.exp(-500*(x**2 + y**2 + z**2))      # extremely peaked center
    tail  = 1 / (1 + x**2 + y**2 + z**2)           # Cauchy-like long tail
    return 0.7*tail + 0.3*spike

def manifold(x, y, z):
    # Points near a curve x = t, y = sin(5t), z = cos(3t)
    h = np.exp(-1 * ((y - np.sin(5*x))**2 + (z - np.cos(3*x))**2))
    return h

# ===================================================================
# 4D Data
# ===================================================================

def metropolis_hastings_4D(unnormalized_density, n_samples, proposal_width=1):
    samples = np.zeros((n_samples, 4))
    
    current_point = np.random.randn(4)
    current_density = unnormalized_density(*current_point) + 1e-9
    
    for i in range(n_samples):
        proposed_point = current_point + proposal_width * np.random.randn(4)
        proposed_density = unnormalized_density(*proposed_point) + 1e-9

        acceptance_ratio = proposed_density / current_density
        
        if np.random.rand() < acceptance_ratio:
            current_point = proposed_point
            current_density = proposed_density
            
        samples[i] = current_point
        
    return samples

def generate_samples_4D(unnormalized_density, n_samples=10000, burn_in=1000, proposal_width=0.5):
    all_samples = metropolis_hastings_4D(unnormalized_density, n_samples + burn_in, proposal_width)
    samples = all_samples[burn_in:]
    return samples

def gauss_pdf_4(x, y, z, w, h=0):
    return np.exp(-1*((x**2 - h)**2 + (y**2 - h)**2 + (z**2 - h)**2 + (w**2 - h)**2))

def heavytail_spike_4(x, y, z, w):
    spike = np.exp(-800*(x**2 + y**2 + z**2 + w**2))      # extremely sharp
    tail  = 1 / (1 + x**2 + y**2 + z**2 + w**2)          # 4D Cauchy-like
    return 0.3*spike + 0.7*tail
    
def heavy_mixture_4(x, y, z, w):
    # Component 1: Sharp Gaussian peaks (localized)
    peak1 = np.exp(-((x - 2)**2 + (y - 2)**2 + (z - 2)**2 + (w - 2)**2) / 0.1)
    peak2 = np.exp(-((x + 2)**2 + (y + 2)**2 + (z + 2)**2 + (w + 2)**2) / 0.1)

    # Component 2: Heavy-tailed (Cauchy-like) component
    tail = 1 / (1 + x**2 + y**2 + z**2 + w**2)
    # Mixture of the two components
    return 0.6 * peak1 + 0.4 * peak2 + 0.3 * tail

