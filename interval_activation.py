
import networkx as nx
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import binom
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 添加中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


stempsDrawCounter = list(range(50, 2000, 50))

N = 1000
p_inner = 0.05
initial_active_count = 40
timesteps = 2001
energy_const = 10000
calculateOverlapNum = 20
G = nx.gnp_random_graph(N, p_inner, directed=True)
initial_active = np.random.choice(N, initial_active_count, replace=False)

def simulate_ca3_network(init_delta, increase=0,interval = 0 , plot=True, save=False, save_prefix="delta_run",):
    activation_per_step = []

    # 初始化状态
    active = np.zeros(N, dtype=bool)
    energy = np.zeros(N)
    activation_counter = np.zeros(N, dtype=int)

    active[initial_active] = True
    topcalculateOverlapNum_nodes_list = []
    delta=init_delta

    for t in range(timesteps):

        received_energy = np.zeros(N)
        n_t = np.sum(active)
        # if n_t == 0:
        #     break

        # 理论阈值计算
        mean_t = energy_const / N * n_t
        var_t = (energy_const / (N * p_inner)) ** 2 * n_t * p_inner * (1 - p_inner)
        std_t = np.sqrt(var_t)
        quantile = 1 - n_t / N
        z_t = np.inf if quantile <= 0 or quantile >= 1 else binom.ppf(quantile, N, p_inner)  # fallback
        th_tm = mean_t + norm.ppf(quantile) * std_t if 0 < quantile < 1 else mean_t
        th_t = th_tm + delta * std_t

        # 能量传播
        for node in np.where(active)[0]:
            out_neighbors = list(G.successors(node))
            if not out_neighbors:
                continue
            energy_per_target = energy_const / len(out_neighbors)
            for target in out_neighbors:
                received_energy[target] += energy_per_target

        new_active = received_energy > th_t


        activation_counter[new_active] += 1
        active = new_active
        energy = np.where(new_active, received_energy, 0.0)
        activation_per_step.append(np.sum(new_active))

        if (t + 1) in stempsDrawCounter:

            plot_activation_distribution_with_binomial(
                activation_counter.copy(), T=t + 1, delta=delta,topcalculateOverlapNum_nodes_list=topcalculateOverlapNum_nodes_list
                ,calculateOverlapNum=calculateOverlapNum
            )
        if interval !=0 and (t + 1)%interval == 0 :
            print(f"timestep={t},now to select new interval_activate neruons")
            interval_active = np.random.choice(N, initial_active_count, replace=False)
            set_init = set(interval_active)
            set_interval = set(initial_active)
            # print(f"[{i}]")
            # print("tN:", sorted(set_tN))
            # print("interval:", sorted(set_interval))
            ratio = len(set_init.intersection(set_interval)) / initial_active_count
            print(f"timestep={t},新一轮迭代与初始激活神经元重叠比率为{ratio}")
            new_active[interval_active] = True
            delta = init_delta
            activation_counter[:] = 0

        delta+=increase
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

    plt.figure(figsize=(15, 8))

    for j, delta in enumerate(deltas):
        for i, increase in enumerate(delta_increase):
            for interval in intervals:
                print("begin to run interval case")
                overlap_ratios = [0]
                overlap_ratio, activation,tN_list_interval = simulate_ca3_network(delta, increase, interval)
                overlap_ratios.extend(overlap_ratio)

                while (len(overlap_ratios) < len(stempsDrawCounter)):
                    overlap_ratios.append(1)

                plt.plot(
                    stempsDrawCounter, overlap_ratios,
                    marker='o', markersize=3, linewidth=0.8,
                    label=f'initial_δ={delta:.3f},inc={increase:.3f},interval={interval}'
                )
    for j, delta in enumerate(deltas):
        for i, increase in enumerate(delta_increase):
            for interval in intervals:
                print("begin to run non-interval case")
                overlap_ratios = [0]
                overlap_ratio, activation,tN_list = simulate_ca3_network(delta, increase, 0)
                overlap_ratios.extend(overlap_ratio)

                while (len(overlap_ratios) < len(stempsDrawCounter)):
                    overlap_ratios.append(1)

                plt.plot(
                    stempsDrawCounter, overlap_ratios,
                    marker='o', markersize=3, linewidth=0.8,
                    label=f'initial_δ={delta:.3f},inc={increase:.3f},interval={0}'
                )
    overlap_ratio_list=[]
    while len(tN_list)<len(tN_list_interval):
        tN_list.append(tN_list[len(tN_list)-1])
    for i in range(len(tN_list_interval) ):
        set_tN = set(tN_list[i])
        set_interval = set(tN_list_interval[i])
        # print(f"[{i}]")
        # print("tN:", sorted(set_tN))
        # print("interval:", sorted(set_interval))
        ratio = len(set_tN.intersection(set_interval)) / calculateOverlapNum
        overlap_ratio_list.append(ratio)
        N = len(overlap_ratio_list)
    plt.plot(
             stempsDrawCounter[:N], overlap_ratio_list,
             marker='o', markersize=3, linewidth=0.8,
        label=f'插入神经元与不插入神经元累计激活最多神经元的重合率'
    )


    plt.xlabel("迭代次数")
    plt.ylabel("神经元重叠比例")
    plt.title("神经元重叠比例随时间变化")
    plt.legend(title=f"连接概率:{p_inner:.2f},交集节点数量:{calculateOverlapNum}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    filename = "overlapratio_all_delta.png"
    plt.savefig(filename, dpi=300)
    print(f"整合图已保存至: {filename}")
    plt.show()

deltas_initial = [0.2]
delta_increase = [0.001]
intervals = [800]
# print(deltas)
plot_overlap_ratios_multi_delta(deltas_initial,delta_increase,intervals)