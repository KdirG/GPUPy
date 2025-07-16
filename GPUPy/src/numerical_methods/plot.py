



def interpolate_and_plot(x, y, x_new, method="spline", bc_type="natural",
                         title=None, save_path=None, show=True):
    """
    Interpolate using either linear or spline method and plot the result.
    
    Args:
        x (array): Known x values (must be strictly increasing)
        y (array): Known y values
        x_new (array): New x values to interpolate
        method (str): Interpolation method: "linear" or "spline"
        bc_type (str): Boundary condition type for spline ("natural", "clamped", "not-a-knot")
        title (str): Custom plot title (optional)
        save_path (str): Path to save the plot (optional)
        show (bool): Whether to display the plot (default: True)
    
    Returns:
        tuple: (interpolated y values, matplotlib figure object)
    """
    if method == "linear":
        interpolator = interp1d(x, y, kind="linear")
        y_new = interpolator(x_new)
        title = title or "Linear Interpolation Results"
    elif method == "spline":
        interpolator = CubicSpline(x, y, bc_type=bc_type)
        y_new = interpolator(x_new)
        title = title or f"Cubic Spline Interpolation ({bc_type})"
    else:
        raise ValueError("Invalid method. Choose 'linear' or 'spline'.")
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_new, y_new, 'b-', linewidth=2, alpha=0.9, label=f'{method.capitalize()} interpolation')
    ax.plot(x, y, 'ro', markersize=8, label='Original data')
    ax.plot(x_new, y_new, 'g.', markersize=5, alpha=0.5, label='Interpolated points')
    
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('x values', fontsize=12)
    ax.set_ylabel('y values', fontsize=12)
    ax.legend(fontsize=10, framealpha=1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()

    return y_new, fig