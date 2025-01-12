import matplotlib.pyplot as plt
import numpy as np
# from analysis.get_surface import get_contour
import matplotlib.patches as patches


def plot_parallel(height=3, cmap="gray", clim=(None, None), v_low=0, v_high=0, pad=0.5,
                  show_axis=False, show_title=False, show_bar=False, show=True,
                  **kwargs):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(kwargs),
        figsize=(height * len(kwargs), height)
    )
    if len(kwargs) == 1:
        axes = [axes]
    for ax, (k, v) in zip(axes, kwargs.items()):
        v = np.array(v, "float32")
        if cmap != "gray":
            pcm = ax.imshow(v, vmax=v_high, cmap=cmap)
        elif v_high != v_low:
            pcm = ax.imshow(v, cmap=cmap, clim=clim, vmin=v_low, vmax=v_high)
        else:
            pcm = ax.imshow(v, cmap=cmap, clim=clim)

        if not show_axis:
            ax.axis("off")
        if show_title:
            ax.set_title(k)
        if show_bar:
            fig.colorbar(pcm, ax=ax)

    fig.tight_layout()
    if show:
        plt.tight_layout(w_pad=pad)
        plt.show()
    else:
        fig.tight_layout(w_pad=pad)


def merge_mask(img, mask, show=True):
    w, h = img.shape
    blue = [250/255, 127/255, 111/255]
    # blue = [1, 0, 0]
    merged = np.zeros([w, h, 3])

    for i in range(3):
        merged[:, :, i] = img * (1 - mask)
        merged[:, :, i] += blue[i] * mask
    if show:
        plt.imshow(merged)
        plt.show()
    return merged


def merge_two_mask(img, mask_1, mask_2, show=False):
    w, h = img.shape
    mask_1 = mask_1 - mask_1 * mask_2
    # red_mask = [242/255, 155/255, 142/255]
    # blue_mask = [103/255, 136/255, 180/255]
    blue_mask = [130/255, 176/255, 210/255]
    red_mask = [250/255, 127/255, 111/255]
    merged = np.zeros([w, h, 3])

    for i in range(3):
        merged[:, :, i] = img * (1 - (mask_1 + mask_2))
        merged[:, :, i] += blue_mask[i] * mask_1 + red_mask[i] * mask_2
    if show:
        plt.imshow(merged)
        plt.show()
    return merged


def merge_two_figure(img_1, img_2):
    assert img_1.shape == img_2.shape
    new_img = np.zeros(img_1.shape)
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            # if i > 512 - j:
            if i > 256:
                new_img[i, j] = img_2[i, j]
            else:
                new_img[i, j] = img_1[i, j]

    return new_img


def merge_mask_edge(img, mask, show=True):
    w, h = img.shape
    blue = [200/255, 36/255, 35/255]
    merged = np.zeros([w, h, 3])
    contour = 1 - get_contour((1 - mask) * 255) / 255
    for i in range(3):
        merged[:, :, i] = img * (1 - contour)
        merged[:, :, i] += blue[i] * contour
    if show:
        plt.imshow(merged)
        plt.show()
    return merged


def save_figure(fig, save_path, cmap="gray", vmin=0, vmax=1):
    if cmap == "gray":
        fig = np.clip(fig, 0, 1)
        # fig = (fig - np.min(fig)) / (np.max(fig) - np.min(fig))
        plt.imshow(fig, cmap="gray", vmin=vmin, vmax=vmax)
    elif cmap == "nature":
        # fig = (fig - np.min(fig)) / (np.max(fig) - np.min(fig))
        fig = np.clip(fig, 0, 1)
        plt.imshow(fig, vmin=vmin, vmax=vmax)
    else:
        fig = (fig - 0) / (np.max(fig) - 0)
        plt.imshow(fig, cmap=cmap)
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
    plt.show()



def merge_img_with_box(img, box):
    points = np.array([(box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3])])
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap="gray")

    polygon = patches.Polygon(points, closed=True, edgecolor="g", linewidth=2, facecolor="none")
    ax.add_patch(polygon)

    plt.show()





















