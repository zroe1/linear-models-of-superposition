import torch
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 100

def create_enhanced_phase_diagram(weights, unembedding_w, bias, device):
    """
    Create a phase diagram showing class predictions across the latent space
    with overlaid weight vectors.
    """
    # Create phase diagram
    print("Generating phase diagram...")
    outputs = torch.zeros((121, 121))
    x_coords = np.arange(-3, 3.05, 0.05)
    y_coords = np.arange(-3, 3.05, 0.05)
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            h = torch.tensor([[x], [y]]).float().to(device)
            out = unembedding_w @ h + bias
            label = torch.argmax(out, dim=0)
            # Fix: j is y-index (row), i is x-index (column)
            outputs[j][i] = label.item()
    
    # Debug: Check what classes are actually predicted
    unique_classes = torch.unique(outputs).int().tolist()
    print(f"Classes predicted by model: {unique_classes}")
    print(f"Total number of unique classes: {len(unique_classes)}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=170)
    
    # Generate NUM_CLASSES unique colors
    def generate_distinct_colors(n):
        """Generate n distinct colors using HSV color space"""
        import matplotlib.colors as mcolors
        colors = []
        for i in range(n):
            hue = i / n  # Evenly space hues around the color wheel
            saturation = 0.7 + 0.3 * (i % 2)  # Alternate between 0.7 and 1.0 saturation
            value = 0.8 + 0.2 * ((i // 2) % 2)  # Alternate brightness for better distinction
            rgb = mcolors.hsv_to_rgb([hue, saturation, value])
            colors.append(mcolors.to_hex(rgb))
        return colors
    
    # Alternative approach using matplotlib's tab20 and other colormaps
    def generate_colors_from_colormaps(n):
        """Generate colors using matplotlib's predefined colormaps"""
        import matplotlib.cm as cm
        if n <= 10:
            return [cm.tab10(i) for i in range(n)]
        elif n <= 20:
            return [cm.tab20(i) for i in range(n)]
        else:
            # For more than 20, use a continuous colormap
            return [cm.hsv(i/n) for i in range(n)]
    
    # Use the HSV approach for better color distinction
    base_colors = generate_distinct_colors(NUM_CLASSES)
    
    # Create lighter versions for the heatmap
    def lighten_color(color, factor=0.6):
        """Make a color lighter by mixing with white"""
        import matplotlib.colors as mcolors
        c = mcolors.to_rgb(color)
        return tuple(c[i] + (1 - c[i]) * factor for i in range(3))
    
    # Create light colors for heatmap - only for the classes that actually appear
    max_class = int(outputs.max())
    light_colors = [lighten_color(base_colors[i]) for i in range(max_class + 1)]
    
    print(f"Color mapping:")
    for i in range(max_class + 1):
        if i in unique_classes:
            print(f"  Class {i}: {base_colors[i]} arrow → light {base_colors[i]} heatmap")
    
    # Create a colormap with only the needed colors
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(light_colors)
    
    im = ax.imshow(outputs, extent=[-3, 3, -3, 3], origin='lower', 
                   alpha=0.8, cmap=custom_cmap, vmin=0, vmax=max_class)
    
    # Plot weight vectors as arrows only
    arrow_length_threshold = 0.15  # Only show arrows for weights with length > threshold
    for i in range(weights.shape[0]):
        w_i = weights[i]
        
        # Convert to numpy for consistent operations
        w_i_np = w_i.detach().cpu().numpy()
        
        # Normalize and scale the weight vector for arrow display
        arrow_scale = 1.5  # Adjust this to make arrows longer/shorter
        arrow_length = np.linalg.norm(w_i_np)
        if arrow_length > arrow_length_threshold:
            # Normalize and scale
            w_normalized = w_i_np / arrow_length * arrow_scale
            
            # Plot arrow from origin in weight direction using base color
            color = base_colors[i]
            ax.arrow(0, 0, w_normalized[0], w_normalized[1],
                    head_width=0.15, head_length=0.15, 
                    fc=color, ec=color, linewidth=3,
                    label=f'Class {i} weight', alpha=0.9)
        else:
            print(f"Skipping arrow for class {i} (weight length {arrow_length:.4f} < threshold {arrow_length_threshold})")
    
    # Add one-hot vector projections
    print("Adding one-hot vector projections...")
    xs = torch.eye(weights.shape[0])  # Create identity matrix for one-hot vectors
    
    for i in range(weights.shape[0]):
        x = xs[i, :]  # Get one-hot vector for class i
        projected_point = (weights.T @ x).detach().cpu().numpy()  # Project to 2D space
        
        # Plot circle at the projected point
        color = base_colors[i]
        ax.scatter(projected_point[0], projected_point[1], 
                  s=80, c=color, marker='o', 
                  edgecolors='black', linewidth=2,
                  label=f'Class {i} one-hot', alpha=0.9, zorder=10)
        
        print(f"Class {i} one-hot projects to: [{projected_point[0]:.3f}, {projected_point[1]:.3f}]")
    
    # Customize the plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_title('Neural Network Phase Diagram with Weight Vectors', fontsize=14, pad=20)
    
    # Add colorbar for phase diagram
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Predicted Class', fontsize=12)
    n_classes = NUM_CLASSES
    cbar.set_ticks(range(n_classes))
    
    # Add legend for weight vectors
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotation explaining the visualization
    textstr = 'Light regions: Predicted class areas\nDark arrows: Weight vector directions\nCircles: One-hot projections'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    # Save the figure to disk before displaying it
    fig.savefig('enhanced_phase_diagram.png', dpi=300)
    plt.show()
    
    return outputs