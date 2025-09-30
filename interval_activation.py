
import networkx as nx
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import binom
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 添加中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


stempsDrawCounter = list(range(50, 2000, 50))

N = 1000
p_inner = 0.01
initial_active_count = 40
timesteps = 2001
energy_const = 10000
calculateOverlapNum = 40
G = nx.gnp_random_graph(N, p_inner, directed=True)
initial_active = np.random.choice(N, initial_active_count, replace=False)
reactivation_gap=500

def simulate_ca3_network(init_delta, increase=0.00 ,interval = 1 , plot=True, save=False, save_prefix="delta_run",):

    activation_per_step = []
    interval = max(interval, 1)  #   //避免interval为0出错
    print("interval = ", interval)
    # 初始化状态
    active = np.zeros(N, dtype=bool)
    energy = np.zeros(N)
    activation_counter = np.zeros(N, dtype=int)

    active[initial_active] = True
    topcalculateOverlapNum_nodes_list = []
    delta=init_delta

    # 初始化计数缓冲区
    buffer = np.zeros((N, interval), dtype=np.int32)
    pos_cache, xy_limits = make_pos_cache(G, seed=42)

    for t in range(timesteps):

        received_energy = np.zeros(N)
        n_t = np.sum(active)
        # if n_t == 0:
        #     break

        # 理论计算阈值的均值
        mean_t = energy_const / N * n_t
        var_t = (energy_const / (N * p_inner)) ** 2 * n_t * p_inner * (1 - p_inner)
        std_t = np.sqrt(var_t)
        quantile = 1 - n_t / N
        z_t = np.inf if quantile <= 0 or quantile >= 1 else binom.ppf(quantile, N, p_inner)  # fallback
        th_tm = mean_t + norm.ppf(quantile) * std_t if 0 < quantile < 1 else mean_t

        #
        slot = t % interval
        prev_slot = (t - 1) % interval
        cum_prev = buffer[:, prev_slot]
        cum_past = buffer[:, slot]
        window_count = cum_prev - cum_past
        # print(cum_past)
        # print(window_count)
        th_vec = th_tm + (delta + window_count*increase) * std_t


        # 能量传播
        for node in np.where(active)[0]:
            out_neighbors = list(G.successors(node))
            if not out_neighbors:
                continue
            energy_per_target = energy_const / len(out_neighbors)
            for target in out_neighbors:
                received_energy[target] += energy_per_target

        new_active = (received_energy > th_vec)
        buffer[:, slot] = cum_prev + new_active.astype(np.int32)
        active = new_active


        activation_counter[new_active] += 1
        activation_per_step.append(np.sum(new_active))

        if (t + 1) in stempsDrawCounter:

            plot_activation_distribution_with_binomial(
                activation_counter.copy(), T=t + 1, delta=delta,topcalculateOverlapNum_nodes_list=topcalculateOverlapNum_nodes_list
                ,calculateOverlapNum=calculateOverlapNum
            )

        # 新挑选神经元进行激活
        if reactivation_gap >1 and (t + 1)%reactivation_gap == 0 :
            print(f"timestep={t},now to select new interval_activate neruons")
            interval_active = np.random.choice(N, initial_active_count, replace=False)
            set_init = set(interval_active)
            set_interval = set(initial_active)
            plot_network_activation_nodes_only(
                G,
                activation_counter.copy(),
                pos=pos_cache,
                xy_limits=xy_limits,
                title=f"Activation map | t={t + 1} (interval={interval})",
                save=True,
                filename=f"{save_prefix}_nodes_t{t + 1}_interval_{interval}.png",
                node_size=18,
                label_top_k=calculateOverlapNum,  # 不标注可读性最高；需要标注再改数字
                vmax_mode="interval",  # 或 "running"/"data"
                interval_len=interval,
                cmap_name="Reds",
            )
            # print(f"[{i}]")
            # print("tN:", sorted(set_tN))
            # print("interval:", sorted(set_interval))
            ratio = len(set_init.intersection(set_interval)) / initial_active_count
            print(f"timestep={t},新一轮迭代与初始激活神经元重叠比率为{ratio}")
            new_active[interval_active] = True
            activation_counter[:] = 0

        # delta+=increase
    # print(activation_per_step)
    overlap_ratios = []
    # 计算 overlap_ratios

    for i in range(len(topcalculateOverlapNum_nodes_list) - 1):
        set_current = set(topcalculateOverlapNum_nodes_list[i])
        set_next = set(topcalculateOverlapNum_nodes_list[i + 1])
        ratio = len(set_current.intersection(set_next)) / calculateOverlapNum
        overlap_ratios.append(ratio)

    # plot_activation_curve(activation_per_step, init_delta,increase,
    #                       save=save,
    #                       filename=f"{save_prefix}_curve_delta_{delta:.2f}.png" if save else None)

    return overlap_ratios,activation_per_step,topcalculateOverlapNum_nodes_list


def make_pos_cache(G, seed=42, margin=0.05):
    pos = nx.spring_layout(G, seed=seed, dim=2, iterations=100)
    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    xr = xs.max() - xs.min()
    yr = ys.max() - ys.min()
    xlim = (xs.min() - margin * xr, xs.max() + margin * xr)
    ylim = (ys.min() - margin * yr, ys.max() + margin * yr)
    return pos, (xlim, ylim)


def plot_network_activation_nodes_only(
    G,
    activation_counter,
    pos,                      # 必须传入：用上面缓存的 pos_cache
    xy_limits=None,           # 可传入：make_pos_cache 返回的 (xlim, ylim)
    title=None,
    save=False,
    filename=None,
    show_colorbar=True,
    node_size=18,
    label_top_k=0,            # 默认不标注；如需标注，设为正整数
    outline_top_k=True,
    vmax_mode="interval",     # "data" | "interval" | "running"
    interval_len=None,
    running_max=None,
    cmap_name="magma",        # ← 使用 magma
):
    """
    只绘节点，按 activation 次数着色（magma）。布局与坐标范围外部固定，保证每张图一致。
    """
    # 1) 准备数据（按 G 的节点顺序）
    node_list = list(G.nodes())
    ac = np.asarray(activation_counter, dtype=float)
    counts = []
    for n in node_list:
        if isinstance(n, (int, np.integer)) and 0 <= n < len(ac):
            counts.append(ac[int(n)])
        else:
            counts.append(0.0)
    counts = np.asarray(counts, dtype=float)

    # 2) 色阶
    vmin = 0.0

    vmax = float(max(counts.max(), 1.0))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 3) 画图（只节点，无边）
    fig, ax = plt.subplots(figsize=(8, 7))
    nodes_artist = nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_color=counts,
        node_size=node_size,
        linewidths=0.0,
        vmin=vmin, vmax=vmax,
        cmap=get_cmap(cmap_name),   # ← magma
        ax=ax,
    )

    # 4) 可选：标注最活跃 Top-K（不排序原顺序不变；这里只挑 K 个最大值的索引）
    if label_top_k and label_top_k > 0:
        top_idx = np.argpartition(-counts, kth=min(label_top_k, len(counts)-1))[:label_top_k]
        for idx in top_idx:
            node_id = node_list[idx]
            x, y = pos[node_id]
            ax.text(
                x, y, f"{node_id}",
                fontsize=7, ha='center', va='center', color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )
        if outline_top_k:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node_list[i] for i in top_idx],
                node_size=node_size + 8,
                node_color='none',
                linewidths=0.9,
                edgecolors='black',
                ax=ax,
            )

    # 5) 颜色条
    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=get_cmap(cmap_name))
        sm.set_array(counts)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Activations in this interval")
        cbar.ax.set_title(f"max={vmax:.0f}", fontsize=8)

    # 6) 固定坐标范围（保证多图位置/比例一致）
    if xy_limits is not None:
        xlim, ylim = xy_limits
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    if save and filename:
        fig.savefig(filename, dpi=180)
    plt.show()
    plt.close(fig)



def plot_activation_distribution_with_binomial(activation_counter, T, delta,topcalculateOverlapNum_nodes_list,calculateOverlapNum):
    topcalculateOverlapNum = np.argsort(activation_counter)[-calculateOverlapNum:][::-1]
    topcalculateOverlapNum_nodes_list.append(topcalculateOverlapNum.tolist())

def plot_activation_curve(activation_per_step, delta,increase, save=False, filename=None):
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, len(activation_per_step) + 1),
        activation_per_step,
        marker='o',
        markersize=2,         # 更小的点
        linewidth=0.8,        # 更细的线
        color='slategray',     # 换掉绿色
        label='新激活神经元数'
    )

    plt.title(f"每轮激活数量变化(init_δ = {delta:.3f},increase = {increase:.3f})")
    plt.xlabel("迭代轮数")
    plt.ylabel("本轮新激活神经元数量")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save and filename:
        plt.savefig(filename, dpi=300)
        print(f"激活曲线图已保存至: {filename}")
        plt.close()
    else:
        plt.show()

def save_activation_plot(activation_counter, T, delta, filename):
    N = len(activation_counter)
    p = np.sum(activation_counter) / (N * T)
    x = np.arange(0, activation_counter.max() + 1)
    binom_probs = binom.pmf(x, T, p)
    expected_counts = N * binom_probs

    plt.figure(figsize=(10, 5))
    plt.hist(activation_counter, bins=np.arange(activation_counter.max() + 2),
             align='left', rwidth=0.8, color='skyblue', edgecolor='black', label='实际')
    plt.plot(x, expected_counts, 'o-', color='red', linewidth=2,
             label=f'理论 B({T}, {p:.3f}) @ δ={delta:.2f}')
    plt.title(f"激活次数分布 vs 二项分布（δ = {delta:.2f}）")
    plt.xlabel("激活次数")
    plt.ylabel("节点频数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"图已保存至: {filename}")

def plot_overlap_ratios_multi_delta(deltas,delta_increase = [0.00], intervals=[0]):

    # plt.figure(figsize=(15, 8))

    # for _, delta in enumerate(deltas):
    #     for _, increase in enumerate(delta_increase):
    #         for interval in intervals:
    #             overlap_ratios = [0]
    #             overlap_ratio, activation,tN_list_interval = simulate_ca3_network(delta, increase, interval)
    #             overlap_ratios.extend(overlap_ratio)
    #
    #             while (len(overlap_ratios) < len(stempsDrawCounter)):
    #                 overlap_ratios.append(1)

                # plt.plot(
                #     stempsDrawCounter, overlap_ratios,
                #     marker='o', markersize=3, linewidth=0.8,
                #     label=f'initial_δ={delta:.3f},inc={increase:.3f},interval={interval}'
                # )
    #
    #             plt.plot(
    #                 list(range(0, timesteps, 1)), activation,
    #                 marker='o', markersize=3, linewidth=0.8,
    #                 label=f'initial_δ={delta:.3f},inc={increase:.3f},interval={interval}'
    #             )

    for j, delta in enumerate(deltas):
        for i, increase in enumerate(delta_increase):
            for interval in intervals:
                print("begin to run non-interval case")
                overlap_ratios = [0]
                overlap_ratio, activation,tN_list = simulate_ca3_network(delta, increase, 1)
                overlap_ratios.extend(overlap_ratio)
    #
    #             while (len(overlap_ratios) < len(stempsDrawCounter)):
    #                 overlap_ratios.append(1)

                # plt.plot(
                #     stempsDrawCounter, overlap_ratios,
                #     marker='o', markersize=3, linewidth=0.8,
                #     label=f'initial_δ={delta:.3f},inc={increase:.3f},interval={0}'
                # )
                # plt.plot(
                #     list(range(0, timesteps, 1)), activation,
                #     marker='o', markersize=3, linewidth=0.8,
                #     label=f'initial_δ={delta:.3f},inc={increase:.3f},interval={interval}'
                # )
    # overlap_ratio_list=[]
    # while len(tN_list)<len(tN_list_interval):
    #     tN_list.append(tN_list[len(tN_list)-1])
    # for i in range(len(tN_list_interval) ):
    #     set_tN = set(tN_list[i])
    #     set_interval = set(tN_list_interval[i])
    #     # print(f"[{i}]")
    #     # print("tN:", sorted(set_tN))
    #     # print("interval:", sorted(set_interval))
    #     ratio = len(set_tN.intersection(set_interval)) / calculateOverlapNum
    #     overlap_ratio_list.append(ratio)
    #     N = len(overlap_ratio_list)
    # plt.plot(
    #          stempsDrawCounter[:N], overlap_ratio_list,
    #          marker='o', markersize=3, linewidth=0.8,
    #     label=f'插入神经元与不插入神经元累计激活最多神经元的重合率'
    # )


    # plt.xlabel("迭代次数")
    # plt.ylabel("神经元重叠比例")
    # plt.title("神经元重叠比例随时间变化")
    # plt.legend(title=f"连接概率:{p_inner:.2f},交集节点数量:{calculateOverlapNum}")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    #
    # filename = "overlapratio_all_delta.png"
    # plt.savefig(filename, dpi=300)
    # print(f"整合图已保存至: {filename}")
    # plt.show()

deltas_initial = [0.2]
delta_increase = [0.1]
intervals = [500]

# print(deltas)
plot_overlap_ratios_multi_delta(deltas_initial,delta_increase,intervals)