
def FFT_2d_heatmap(ax, fig, VX, VY, Intensity, title, cmap='gray'):
    c = ax.imshow(Intensity, extent=[VX.min(), VX.max(), VY.min(), VY.max()],
               origin='lower', cmap=cmap)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Frequency $\\nu_x$')
    ax.set_ylabel('Frequency $\\nu_y$')
    ax.set_title(title)
    return ax

def XY_2d_heatmap(ax, fig, x, y, Intensity, title, cmap='gray'):
    c = ax.imshow(Intensity, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap=cmap)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return ax