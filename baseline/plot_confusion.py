import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))  # , normalize=normalize) 部分版本无该参数
    return cmtx


def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


from torch.utils.tensorboard import SummaryWriter


def add_confusion_matrix(
        writer,
        cmtx,
        num_classes,
        global_step=None,
        subset_ids=None,
        class_names=None,
        tag="Confusion Matrix",
        figsize=None,
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
        )
        # Add the confusion matrix image to writer.
        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)


def plot_xSec_yMv(datas, fs, ceiling=4, duration=10, title=None, save_path=None):
    if fs != 250:
        print('需要重采样')
        # ecg_datas = resample_datas(datas, fs, 250)
        ecg_datas = re_datas(datas, fs, 250)
    else:
        ecg_datas = [range(len(datas)), datas]

    fig_with = (duration / 0.04) / 10
    print('fig_with', fig_with)
    fig_height = (ceiling * 2 / 0.1) / 10
    print('fig_height', fig_height)
    fig = plt.figure(figsize=(fig_with, fig_height))
    axes = fig.add_subplot(1, 1, 1)

    font = FontProperties(fname="../so/msyh.ttc", size=14)  # 设置字体
    axes.set_title(title, fontproperties=font)

    xlim_end = int(fig_with * (50 / 0.5))
    axes.set_xlim(0, xlim_end)
    axes.set_ylim(-ceiling, ceiling)

    # 大格
    big_with = 50
    print('big_with', big_with)
    big_height = 0.5
    big_miloc_x = plt.MultipleLocator(big_with)
    # big_miloc_x.MAXTICKS = 50000
    big_miloc_y = plt.MultipleLocator(big_height)
    axes.xaxis.set_minor_locator(big_miloc_x)
    axes.yaxis.set_minor_locator(big_miloc_y)
    axes.grid(axis='both', which='major', c='r', ls='-', lw='0.5')

    # 小格
    small_with = 10
    print('small_with', small_with)
    small_height = 0.1
    small_miloc_x = plt.MultipleLocator(small_with)
    # small_miloc_x.MAXTICKS = 50000
    small_miloc_y = plt.MultipleLocator(small_height)
    axes.xaxis.set_minor_locator(small_miloc_x)
    axes.yaxis.set_minor_locator(small_miloc_y)
    axes.grid(axis='both', which='minor', c='#ff7855', ls=':', lw='0.5')

    # 刻度
    _xticks = set()
    big_size = xlim_end // big_with
    for i in range(big_size):
        _xticks.add(i * big_with)
    xticks = sorted(_xticks)
    xticks.append(xticks[-1] + big_with)
    axes.set_xticks(xticks)
    _yticks = set()
    for i in range(int(ceiling / 0.5)):
        _yticks.add(i * 0.5)
        _yticks.add(-i * 0.5)
    yticks = sorted(_yticks)
    yticks.append(yticks[-1] + 0.5)
    axes.set_yticks(yticks)

    # 刻度标签
    axes.set_xticklabels(xticks)
    axes.set_yticklabels(['%smv' % _yt for _yt in yticks])

    axes.plot(ecg_datas[0], ecg_datas[1], c='k', lw='0.8')
    # axes.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.clf()
    plt.close(fig)