import matplotlib.pyplot as plt

# 绘制心电图
# datas 心电毫伏值数据
# fs 采样率
# ceiling Y轴心电毫伏值上下限展示范围, 单位毫伏
# duration X轴心电持续时长, 单位秒
# title 标题
# save_path 保存路径,如F:/demo.png
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

    # font = FontProperties(fname="../so/msyh.ttc", size=14)  # 设置字体
    axes.set_title(title)

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


# 绘制心电图-两个子图
# datas 心电毫伏值数据, datas1 datas2
# fs 采样率, fs1 fs2
# ceiling Y轴心电毫伏值上下限展示范围, 单位毫伏
# duration X轴心电持续时长, 单位秒
# title 标题
# save_path 保存路径,如F:/demo.png
def plot_xSec_yMv_2(datas1, datas2, fs1, fs2, ceiling=4, duration=10, title=None, save_path=None):
    if fs1 != 250:
        print('datas1需要重采样')
        # ecg_datas1 = resample_datas(datas1, fs1, 250)
        ecg_datas1 = re_datas(datas1, fs1, 250)
    else:
        ecg_datas1 = [range(len(datas1)), datas1]

    if fs2 != 250:
        print('datas2需要重采样')
        # ecg_datas2 = resample_datas(datas2, fs2, 250)
        ecg_datas2 = re_datas(datas2, fs2, 250)
    else:
        ecg_datas2 = [range(len(datas2)), datas2]

    fig_with = (duration / 0.04) / 10
    # print('fig_with', fig_with)
    fig_height = (ceiling * 4 / 0.1) / 10
    # print('fig_height', fig_height)
    fig = plt.figure(figsize=(fig_with, fig_height))
    axes1 = fig.add_subplot(2, 1, 1)
    axes2 = fig.add_subplot(2, 1, 2)
    fig.subplots_adjust(wspace=0, hspace=0)

    # font = FontProperties(fname="../so/msyh.ttc", size=14)  # 设置字体
    axes1.set_title(title)
    # axes2.set_title(title, fontproperties=font)

    xlim_end = int(fig_with * (50 / 0.5))
    axes1.set_xlim(0, xlim_end)
    axes2.set_xlim(0, xlim_end)
    axes1.set_ylim(-ceiling, ceiling)
    axes2.set_ylim(-ceiling, ceiling)

    # 大格
    big_with = 50
    # print('big_with', big_with)
    big_height = 0.5
    big_miloc_x = plt.MultipleLocator(big_with)
    # big_miloc_x.MAXTICKS = 50000
    big_miloc_y = plt.MultipleLocator(big_height)
    axes1.xaxis.set_minor_locator(big_miloc_x)
    axes2.xaxis.set_minor_locator(big_miloc_x)
    axes1.yaxis.set_minor_locator(big_miloc_y)
    axes2.yaxis.set_minor_locator(big_miloc_y)
    axes1.grid(axis='both', which='major', c='r', ls='-', lw='0.5')
    axes2.grid(axis='both', which='major', c='r', ls='-', lw='0.5')

    # 小格
    small_with = 10
    # print('small_with', small_with)
    small_height = 0.1
    small_miloc_x = plt.MultipleLocator(small_with)
    # small_miloc_x.MAXTICKS = 50000
    small_miloc_y = plt.MultipleLocator(small_height)
    axes1.xaxis.set_minor_locator(small_miloc_x)
    axes2.xaxis.set_minor_locator(small_miloc_x)
    axes1.yaxis.set_minor_locator(small_miloc_y)
    axes2.yaxis.set_minor_locator(small_miloc_y)
    axes1.grid(axis='both', which='minor', c='#ff7855', ls=':', lw='0.5')
    axes2.grid(axis='both', which='minor', c='#ff7855', ls=':', lw='0.5')

    # 刻度
    _xticks = set()
    big_size = xlim_end // big_with
    for i in range(big_size):
        _xticks.add(i * big_with)
    xticks = sorted(_xticks)
    xticks.append(xticks[-1] + big_with)
    axes1.set_xticks(xticks)
    axes2.set_xticks(xticks)
    _yticks = set()
    for i in range(int(ceiling / 0.5)):
        _yticks.add(i * 0.5)
        _yticks.add(-i * 0.5)
    yticks = sorted(_yticks)
    yticks.append(yticks[-1] + 0.5)
    axes1.set_yticks(yticks)
    axes2.set_yticks(yticks)

    # 刻度标签
    axes1.set_xticklabels(['' for _ in xticks])
    axes2.set_xticklabels(xticks)
    axes1.set_yticklabels(['%smv' % _yt for _yt in yticks])
    axes2.set_yticklabels(['%smv' % _yt for _yt in yticks])

    axes1.plot(ecg_datas1[0], ecg_datas1[1], c='k', lw='0.8')
    axes2.plot(ecg_datas2[0], ecg_datas2[1], c='k', lw='0.8')
    # axes1.legend()
    # axes2.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.clf()
    plt.close(fig)