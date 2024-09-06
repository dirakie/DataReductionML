import os
import matplotlib.pyplot as plt

# this function stores figures in the desired chapters figures directory.
# example usage: save_figure("results", "random_figure2", "png")
def save_figure(chapter, fig_name, format):
    
    # create directory if not existent
    dir_path = os.path.join(".", chapter, "figures")
    os.makedirs(dir_path, exist_ok=True)

    # save figure
    fig_path = os.path.join(dir_path, fig_name + "." + str(format))
    plt.tight_layout()
    plt.savefig(fig_path, format=format, dpi=300)
    print(f"Successfully saved figure {fig_name} as .{format} file to {dir_path}")