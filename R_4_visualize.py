import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_results_nd(points, density, grid_shape, bounds, bandwidth, loss_history, name='1'):
    N_dims = len(grid_shape)
    
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(density, torch.Tensor):
        density = density.cpu().numpy()

    if N_dims == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        axes[0].plot(loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        # axes[0].set_title('Optimization Progress')
        axes[0].grid(True, alpha=0.3)
        
        # Density plot
        x_vals = np.linspace(bounds[0][0], bounds[0][1], grid_shape[0])
        axes[1].plot(x_vals, density, linewidth=2, label=f'$h$ = {bandwidth:.4f}')
        axes[1].scatter(points, np.zeros_like(points), 
                        alpha=0.5, s=20, c='red', label='Data')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Density')
        # axes[1].set_title(f'TDA KDE (h={bandwidth:.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    elif N_dims == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        axes[0].plot(loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        # axes[0].set_title('Optimization Progress')
        axes[0].grid(True, alpha=0.3)
        
        # Density map
        extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
        im = axes[1].imshow(density.T, origin='lower', 
                            extent=extent, cmap='viridis', aspect='auto')
        axes[1].scatter(points[:, 0], points[:, 1], 
                        c='white', s=1, alpha=0.1, edgecolors='black', linewidths=0.5)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_title(f'$h$ = {bandwidth:.4f}')
        plt.colorbar(im, ax=axes[1], label='Density')
        
    else:  # 3D or higher
        n_proj = min(3, N_dims * (N_dims - 1) // 2) 
        fig, axes = plt.subplots(1, n_proj + 1, figsize=(4 * (n_proj + 1), 4))
        
        # Loss curve
        axes[0].plot(loss_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        # axes[0].set_title('Optimization Progress')
        axes[0].grid(True, alpha=0.3)
        
        # 2D projections
        proj_idx = 1
        for i in range(N_dims):
            for j in range(i + 1, N_dims):
                if proj_idx > n_proj: break
                axes[proj_idx].scatter(points[:, i], points[:, j], alpha=0.6, s=20)
                axes[proj_idx].set_xlabel(f'$X_{i+1}$')
                axes[proj_idx].set_ylabel(f'$X_{j+1}$')
                axes[proj_idx].set_title(f'Projection: dims {i+1}, {j+1}')
                axes[proj_idx].grid(True, alpha=0.3)
                proj_idx += 1
            if proj_idx > n_proj: break
            
    plt.tight_layout()
    plt.savefig(f'experiment_{name}.png', dpi=150)
    plt.show()
    # plt.close(fig) # Close figure to save memory

def plot_comparison_results(Z_kdes, Z_true_norm, linspaces, bandwidths, data_nd=None, name_='comparison'):
    """
    Visualizes the comparison of KDE methods for 1D and 2D.
    """
    D = len(linspaces)
    
    if D == 1:
        x_grid = linspaces[0]
        plt.figure(figsize=(10, 6))
        
        # Plot True PDF
        plt.plot(x_grid, Z_true_norm, 'k--', linewidth=2, label='True PDF')
        
        # Plot all KDEs
        for name, Z_kde in Z_kdes.items():
            if name == 'TDA':
                plt.plot(x_grid, Z_kde, label=name+f' ($h$: {bandwidths[name]:.3f})', linewidth=2.5)
            else:
                plt.plot(x_grid, Z_kde, label=name+f' ($h$: {bandwidths[name]:.3f})', linewidth=1, alpha=0.7)

        plt.xlabel('X')
        plt.ylabel('Density')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.6)
        
    elif D == 2:
        x_grid = linspaces[0]
        y_grid = linspaces[1]
        extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
        
        all_pdfs = {'True PDF': Z_true_norm, **Z_kdes}

        if 'ISJ' in all_pdfs:
            del all_pdfs['ISJ']

        if 'Silverman' in all_pdfs:
            del all_pdfs['Silverman']
        
        n_plots = len(all_pdfs)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        axes = axes.ravel()
        
        for i, (name, Z_pdf) in enumerate(all_pdfs.items()):
            ax = axes[i]
            im = ax.imshow(Z_pdf.T, origin='lower', extent=extent, 
                           aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax)
            
            if data_nd is not None:
                ax.scatter(data_nd[:, 0], data_nd[:, 1], c='white', s=1, alpha=0.1)

            if name == 'True PDF':
                title = name
            else:
                title = f"{name} ($h$: {bandwidths[name]:.3f})"   

            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        # plt.suptitle('2D KDE Comparison', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    else:
        print(f"\n📈 Plotting for D={D} is not supported. Skipping comparison plot.")
        return

    plt.savefig(f'comparison_{name_}.png', dpi=150)
    plt.show()
    # plt.close() # Close figure to save memory