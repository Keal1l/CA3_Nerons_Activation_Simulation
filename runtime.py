import networkx as nx
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import binom, chisquare
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 参数设置
N = 1000
p_inner = 0.01
initial_active_count = 40
timesteps = 1000
energy_const = 10000  # 固定传播能量
delta = 0.15           # 阈值调整参数
stempsDrawCounter=list(range(50, 1000, 50))
calculateOverlapNum = 20

# 迭代激活计数器
activation_counter = np.zeros(N, dtype=int)

# 构建 ER 有向图
G = nx.gnp_random_graph(N, p_inner, directed=True)

# 计算入度和出度
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())
# 按照入度降序排列，取前 calculateOverlapNum 个和后 calculateOverlapNum 个
sorted_in_deg = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
topcalculateOverlapNum_in = sorted_in_deg[:calculateOverlapNum]
bottomcalculateOverlapNum_in = sorted_in_deg[-calculateOverlapNum:]

# 按照出度降序排列，取前 calculateOverlapNum 个和后 calculateOverlapNum 个
sorted_out_deg = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
topcalculateOverlapNum_out = sorted_out_deg[:calculateOverlapNum]
bottomcalculateOverlapNum_out = sorted_out_deg[-calculateOverlapNum:]

def calculate_top_in_and_out(G):
    print("\n入度最多的 calculateOverlapNum 个节点：")
    for node, deg in topcalculateOverlapNum_in:
        print(f"节点 {node}，入度: {deg}")

    print("\n入度最少的 calculateOverlapNum 个节点：")
    for node, deg in bottomcalculateOverlapNum_in:
        print(f"节点 {node}，入度: {deg}")

    print("\n出度最多的 calculateOverlapNum 个节点：")
    for node, deg in topcalculateOverlapNum_out:
        print(f"节点 {node}，出度: {deg}")

    print("\n出度最少的 calculateOverlapNum 个节点：")
    for node, deg in bottomcalculateOverlapNum_out:
        print(f"节点 {node}，出度: {deg}")

def calculate_outdegree(G):
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    avg_out_degree = np.mean(out_degrees)
    print(f"图中节点的平均出度: {avg_out_degree:.2f}")

# 初始化激活状态
active = np.zeros(N, dtype=bool)
energy = np.zeros(N)

# 初始激活神经元（calculateOverlapNum个）
initial_active = np.random.choice(N, initial_active_count, replace=False)
active[initial_active] = True

# 记录经验方差 > 理论方差 的轮数
# count_empirical_gt_theoretical = 0

print(f"初始激活神经元编号: {initial_active.tolist()}")

# 激活历史记录
activation_history = []
topcalculateOverlapNum_nodes_list = []


def plot_activation_distribution_with_binomial(activation_counter, T, delta):
    print(f"\nTime step {t + 1}")
    N = len(activation_counter)
    total_activations = np.sum(activation_counter)
    p = total_activations / (N * T)

    print(f"平均每轮每个神经元激活概率 p = {p:.4f}，对应 delta = {delta:.2f}")

    max_activation = activation_counter.max()
    bins = np.arange(0, max_activation + 2)

    plt.figure(figsize=(10, 5))
    counts, _, _ = plt.hist(activation_counter, bins=bins, align='left',
                            rwidth=0.8, color='skyblue', edgecolor='black',
                            label='实际激活次数直方图')

    # 理论二项分布
    x = np.arange(0, max_activation + 1)
    binom_probs = binom.pmf(x, T, p)
    expected_counts = N * binom_probs

    plt.plot(x, expected_counts, 'o-', color='red', linewidth=2,
             label=f'理论分布 B({T}, {p:.3f}) @ δ={delta:.2f}')

    plt.title(f"神经元激活次数分布（δ = {delta:.2f}）")
    plt.xlabel("节点被激活的次数")
    plt.ylabel("节点数量（频率）")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()
    filename = f"activation_histogram_delta_{delta:.2f}_t_{T}.png"
    # plt.savefig(filename, dpi=300)
    print(f"图已保存至: {filename}")
    plt.close()
    # 提取激活最多的前 calculateOverlapNum 个节点索引，保存到全局列表
    topcalculateOverlapNum = np.argsort(activation_counter)[-calculateOverlapNum:][::-1]
    topcalculateOverlapNum_nodes_list.append(topcalculateOverlapNum.tolist())

    # counts = np.bincount(activation_counter)  # index = 激活次数, value = 出现次数
    # x = np.arange(len(counts))

    # # 第二步：计算理论二项分布的期望频数
    # expected = binom.pmf(x, T, p) * N  # 每个激活次数下理论应有多少神经元
    #
    # # 第三步：为了满足卡方检验条件，只保留 expected > 5 的部分
    # valid = expected > 5
    # counts_valid = counts[valid]
    # expected_valid = expected[valid]

    # 保证理论期望和观察频数总和相等
    # expected_valid = expected_valid * (np.sum(counts_valid) / np.sum(expected_valid))

    # 执行卡方检验
    # chi2_stat, p_val = chisquare(counts_valid, f_exp=expected_valid)
    #
    #
    # print(f"卡方统计量: {chi2_stat:.2f}")
    # print(f"p-value: {p_val:.4e}")
    #
    # if p_val < 0.05:
    #     print("❌ 拒绝原假设：激活次数不符合二项分布 B(T, p)")
    # else:
    #     print("✅ 接受原假设：激活次数符合二项分布 B(T, p)")

# 多步传播模拟
for t in range(timesteps):
    # print(f"\nTime step {t + 1}")

    received_energy = np.zeros(N)
    n_t = np.sum(active)
    if n_t == 0:
        print("无激活神经元，传播终止。")
        break

    # Step 1: 计算动态阈值 th_t
    mean_t = energy_const / N * n_t
    var_t = (energy_const / (N * p_inner)) ** 2 * n_t * p_inner * (1 - p_inner)
    std_t = np.sqrt(var_t)
    quantile = 1 - n_t / N
    z_t = norm.ppf(quantile) if 0 < quantile < 1 else np.inf
    th_tm = mean_t + z_t * std_t
    th_t = th_tm + delta * std_t

    # print("---- 阈值计算详情 ----")
    # print(f"当前激活神经元数量 n_t = {n_t}")
    # print(f"mean_t = {mean_t:.4f}")
    # print(f"std_t = {std_t:.4f}")
    # print(f"quantile = {quantile:.4f}")
    # print(f"z_t = {z_t:.4f}")
    # print(f"th_tm（理论阈值）= {th_tm:.4f}")
    # print(f"th_t（最终阈值）= {th_t:.4f}")
    # print(f"动态期望阈值 th_tm: {th_tm:.2f}，调整后阈值 th_t: {th_t:.2f}（delta = {delta:.2f}）")

    # Step 2: 激活神经元传播能量
    for node in np.where(active)[0]:
        out_neighbors = list(G.successors(node))
        if not out_neighbors:
            print(f"神经元 {node} 被激活，但没有出边，不传播。")
            continue

        energy_per_target = energy_const / len(out_neighbors)
        # print(f"神经元 {node} 被激活，向 {len(out_neighbors)} 个神经元传播 {energy_const} 能量，"
        #       f"每个接收 {energy_per_target:.2f}")

        for target in out_neighbors:
            received_energy[target] += energy_per_target
            # print(f"  -> 向神经元 {target} 传播 {energy_per_target:.2f} 能量")

    nonzero_energy = received_energy[received_energy > 0]
    empirical_mean = np.mean(nonzero_energy) if len(nonzero_energy) > 0 else 0
    empirical_var = np.var(nonzero_energy) if len(nonzero_energy) > 0 else 0
    empirical_std = np.sqrt(empirical_var)

    # print("---- 经验分布统计 ----")
    # print(f"非零接收能量神经元数: {len(nonzero_energy)}")
    # print(f"经验均值: {empirical_mean:.4f}")
    # print(f"经验标准差: {empirical_std:.4f}")
    # print(f"经验方差: {empirical_var:.4f}")
    # if empirical_var < var_t:
    #     count_empirical_gt_theoretical += 1
    #     print(f" 第 {t + 1} 轮：经验方差 ({empirical_var:.4f}) < 理论方差 ({var_t:.4f})")
    # th_tm = empirical_mean + z_t * empirical_std
    # th_t = th_tm + delta * empirical_std

    # 判断新激活神经元
    new_active = received_energy > th_t
    new_indices = np.where(new_active)[0]
    num_new = len(new_indices)
    activation_history.append(num_new)
    activation_counter[new_active] += 1

    # print(f"本轮新激活神经元数量: {num_new}")
    # print("新激活神经元编号:", new_indices.tolist())

    if (t + 1) in stempsDrawCounter:
        plot_activation_distribution_with_binomial(
            activation_counter.copy(), T=t + 1, delta=delta,
        )

    active = new_active
    energy = np.where(new_active, received_energy, 0.0)

# 计算t和t-1时刻累计激活神经元数量的重合比例
# for i in range(len(topcalculateOverlapNum_nodes_list) - 1):
#     set_current = set(topcalculateOverlapNum_nodes_list[i])
#     set_next = set(topcalculateOverlapNum_nodes_list[i + 1])
#
#     intersection = set_current.intersection(set_next)
#     overlap_ratio = len(intersection) / len(set_current)
#
#     print(f"\n第 {i} 次和第 {i+1} 次 overlap:")
#     print(f"  - 重叠数量: {len(intersection)}")
#     print(f"  - 重叠比例: {overlap_ratio:.2f}")
#     print(f"  - 具体重叠节点: {sorted(list(intersection))}")





# 先把这些节点编号单独提取出来（只要编号，不要度数）
topcalculateOverlapNum_in_nodes = [node for node, _ in topcalculateOverlapNum_in]
bottomcalculateOverlapNum_in_nodes = [node for node, _ in bottomcalculateOverlapNum_in]
topcalculateOverlapNum_out_nodes = [node for node, _ in topcalculateOverlapNum_out]
bottomcalculateOverlapNum_out_nodes = [node for node, _ in bottomcalculateOverlapNum_out]

# topcalculateOverlapNum_nodes_list 中最后一次的 topcalculateOverlapNum 集合
last_topcalculateOverlapNum_nodes = set(topcalculateOverlapNum_nodes_list[-1])

# 定义一个函数，计算交集数量和比例
def calc_overlap(set1, set2):
    intersection = set1.intersection(set2)
    overlap_ratio = len(intersection) / len(set1)
    return len(intersection), overlap_ratio, sorted(list(intersection))

# 比较入度最多的
# n_overlap, ratio, overlap_nodes = calc_overlap(set(last_topcalculateOverlapNum_nodes), set(topcalculateOverlapNum_in_nodes))
# print("\n和 入度最多 calculateOverlapNum 节点 重叠:")
# print(f"  - 重叠数量: {n_overlap}")
# print(f"  - 重叠比例: {ratio:.2f}")
# print(f"  - 重叠节点: {overlap_nodes}")
#
# # 比较入度最少的
# n_overlap, ratio, overlap_nodes = calc_overlap(set(last_topcalculateOverlapNum_nodes), set(bottomcalculateOverlapNum_in_nodes))
# print("\n和 入度最少 calculateOverlapNum 节点 重叠:")
# print(f"  - 重叠数量: {n_overlap}")
# print(f"  - 重叠比例: {ratio:.2f}")
# print(f"  - 重叠节点: {overlap_nodes}")
#
# # 比较出度最多的
# n_overlap, ratio, overlap_nodes = calc_overlap(set(last_topcalculateOverlapNum_nodes), set(topcalculateOverlapNum_out_nodes))
# print("\n和 出度最多 calculateOverlapNum 节点 重叠:")
# print(f"  - 重叠数量: {n_overlap}")
# print(f"  - 重叠比例: {ratio:.2f}")
# print(f"  - 重叠节点: {overlap_nodes}")
#
# # 比较出度最少的
# n_overlap, ratio, overlap_nodes = calc_overlap(set(last_topcalculateOverlapNum_nodes), set(bottomcalculateOverlapNum_out_nodes))
# print("\n和 出度最少 calculateOverlapNum 节点 重叠:")
# print(f"  - 重叠数量: {n_overlap}")
# print(f"  - 重叠比例: {ratio:.2f}")
# print(f"  - 重叠节点: {overlap_nodes}")



# plot_activation_histogram(activation_counter, initial_active)


# plot_activation_distribution_with_binomial(activation_counter, timesteps, delta)
# print("\n经验方差小于理论方差的轮数:", count_empirical_gt_theoretical)
# # 输出最终统计结果
# print("\n最终激活神经元总数：", np.sum(active))
# print("各时间步激活数量序列：", activation_history)

def plot_overlap_ratio_comparison(stempsDrawCounter,
                                  topcalculateOverlapNum_nodes_list,
                                  top_in, bottom_in, top_out, bottom_out,
                                  calculateOverlapNum):
    """
    参数：
    - stempsDrawCounter: List[int] 时间点
    - topcalculateOverlapNum_nodes_list: List[List[int]] 各时间点 top20 激活神经元
    - top_in/bottom_in/top_out/bottom_out: List[Tuple[int, int]] (node, degree)
    - calculateOverlapNum: int，每个度数集合的长度
    """
    # 取节点编号
    top_in_nodes = set(node for node, _ in top_in)
    bottom_in_nodes = set(node for node, _ in bottom_in)
    top_out_nodes = set(node for node, _ in top_out)
    bottom_out_nodes = set(node for node, _ in bottom_out)

    # 记录 4 种重叠比例随时间的变化
    overlap_ratios_top_in = []
    overlap_ratios_bottom_in = []
    overlap_ratios_top_out = []
    overlap_ratios_bottom_out = []

    for top_nodes in topcalculateOverlapNum_nodes_list:
        active_nodes = set(top_nodes)

        # 计算 overlap ratio
        def calc_ratio(node_set):
            intersection = active_nodes.intersection(node_set)
            return len(intersection) / calculateOverlapNum

        # overlap_ratios_top_in.append(calc_ratio(top_in_nodes))
        # overlap_ratios_bottom_in.append(calc_ratio(bottom_in_nodes))
        # overlap_ratios_top_out.append(calc_ratio(top_out_nodes))
        # overlap_ratios_bottom_out.append(calc_ratio(bottom_out_nodes))

    # 画图
    # plt.figure(figsize=(10, 6))
    # plt.plot(stempsDrawCounter, overlap_ratios_top_in, marker='o', markersize=3, linewidth=0.8,
    #          label='入度最多')
    # plt.plot(stempsDrawCounter, overlap_ratios_bottom_in, marker='s', markersize=3, linewidth=0.8,
    #          label='入度最少')
    # plt.plot(stempsDrawCounter, overlap_ratios_top_out, marker='^', markersize=3, linewidth=0.8,
    #          label='出度最多')
    # plt.plot(stempsDrawCounter, overlap_ratios_bottom_out, marker='v', markersize=3, linewidth=0.8,
    #          label='出度最少')
    #
    # plt.xlabel("时间点")
    # plt.ylabel("重叠比例")
    # plt.title(f"入度/出度最多/最少的 {calculateOverlapNum} 个节点与活跃 {calculateOverlapNum} 个神经元的重叠比例")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    #
    # filename = "overlap_ratio_comparison_with_topIndegreeNodes.png"
    # plt.savefig(filename, dpi=300)
    # print(f"图已保存至: {filename}")
    # plt.show()

plot_overlap_ratio_comparison(
    stempsDrawCounter,
    topcalculateOverlapNum_nodes_list,
    topcalculateOverlapNum_in,
    bottomcalculateOverlapNum_in,
    topcalculateOverlapNum_out,
    bottomcalculateOverlapNum_out,
    calculateOverlapNum
)