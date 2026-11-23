"""约束求解器：找出每笔票款在两套分组合计约束下仍可行的候选金额。

约束模型
--------
- 每笔票款 i 的取值必须来自其候选集合 domains[i]；
- 每个窗口分组 g 内各笔取值之和必须等于给定合计 totals[g]；
- 每个售票日分组 h 内各笔取值之和必须等于给定合计 totals[h]。

求解方式
--------
单趟带回溯的记忆化搜索（DFS + memo），按笔依次赋值并做前向检查：
不逐一枚举整批填法，而是只记录“某笔取某个值”是否出现在至少一个可行
完整解中，最后返回每笔的可行值集合。剪枝手段：

1. 每个分组剩余笔数的可达金额集合（suffix reachable sets），判断当前
   已用额度 + 剩余笔数能否凑出目标值，不行立即回退；
2. 剩余笔数的最小/最大金额和兜底剪枝；
3. 记忆化状态 (i, 各分组当前已用额度)，合并子搜索中出现过的可行取值。

批量勾选确认用 select_max_adoptions：在同一套约束下把“勾选金额”当作
可采纳/可不采纳的软约束，按行序回溯并记忆化后缀最优（先比采纳笔数、
再按行序比较取舍），一次搜索得出最多能同时成立的勾选。两次搜索共用
SearchBudget，累计状态数不超过原单次求解上限。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


class SolverError(Exception):
    """求解过程中的参数或状态错误。"""


class SearchLimitExceeded(SolverError):
    """搜索状态数超过保护上限，无法在预算内完成判定。"""


class SearchBudget:
    """一次操作内多次搜索共用的状态数预算。

    批量校对会先判定“最多能同时采纳哪些勾选”，再重算一次全表可行金额；
    两次搜索共用同一个预算对象，累计访问状态数不得超过原单次求解的上限。
    """

    __slots__ = ("limit", "used")

    def __init__(self, limit: int = 2_000_000) -> None:
        if limit < 1:
            raise SolverError("搜索状态数上限必须为正整数")
        self.limit = limit
        self.used = 0

    def step(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise SearchLimitExceeded(
                f"搜索状态数超过上限（{self.limit}），请缩小批次或减少勾选"
            )


class _Problem:
    """求解器共用的规范化输入与剪枝索引。"""

    __slots__ = (
        "n",
        "d",
        "gm",
        "t",
        "group_count",
        "members",
        "suffix_pos_after",
        "reach_sets",
        "exact",
        "suffix_min",
        "suffix_max",
    )


def _prepare(
    domains: Sequence[Iterable[int]],
    group_memberships: Sequence[Sequence[int]],
    totals: Sequence[int],
) -> _Problem:
    """校验并规范化输入，预计算分组可达集合与上下界剪枝表。"""
    n = len(domains)
    d = tuple(tuple(sorted(set(vals))) for vals in domains)
    gm = tuple(tuple(gs) for gs in group_memberships)
    t = tuple(totals)

    if n == 0:
        raise SolverError("没有任何票款记录")
    if len(gm) != n:
        raise SolverError("票款与分组归属数量不一致")
    if any(not vals for vals in d):
        raise SolverError("存在没有候选金额的票款记录")
    if any(len(g) != len(gm[0]) for g in gm):
        raise SolverError("每笔票款必须归属相同数量的分组")
    if len(gm[0]) == 0:
        raise SolverError("至少需要一套分组合计约束")
    max_group_id = max(max(g) for g in gm)
    if min(min(g) for g in gm) < 0:
        raise SolverError("分组 id 不能为负")
    group_count = max_group_id + 1
    if len(t) != group_count:
        raise SolverError("合计数量与分组数量不一致")
    if any(total < 0 for total in t):
        raise SolverError("分组合计不能为负")

    members: list[list[int]] = [[] for _ in range(group_count)]
    for i, gs in enumerate(gm):
        for g in gs:
            if g < 0 or g >= group_count:
                raise SolverError(f"第 {i + 1} 笔票款引用了不存在的分组 {g}")
            members[g].append(i)
    for g, ids in enumerate(members):
        if not ids:
            raise SolverError(f"分组 {g} 没有任何票款记录")

    position_in_group: list[list[int]] = [[-1] * n for _ in range(group_count)]
    for g, ids in enumerate(members):
        for pos, i in enumerate(ids):
            position_in_group[g][i] = pos

    suffix_pos_after: list[list[int]] = [[0] * n for _ in range(group_count)]
    for g, ids in enumerate(members):
        for i in range(n):
            p = position_in_group[g][i]
            suffix_pos_after[g][i] = 0 if p == -1 else p + 1

    reach_sets, exact = _group_reachable_sets(
        d, tuple(tuple(ids) for ids in members), t
    )

    suffix_min: list[list[int]] = []
    suffix_max: list[list[int]] = []
    for g, ids in enumerate(members):
        smin = [0] * (len(ids) + 1)
        smax = [0] * (len(ids) + 1)
        for k in range(len(ids) - 1, -1, -1):
            vals = d[ids[k]]
            smin[k] = smin[k + 1] + vals[0]
            smax[k] = smax[k + 1] + vals[-1]
        suffix_min.append(smin)
        suffix_max.append(smax)

    p = _Problem()
    p.n = n
    p.d = d
    p.gm = gm
    p.t = t
    p.group_count = group_count
    p.members = members
    p.suffix_pos_after = suffix_pos_after
    p.reach_sets = reach_sets
    p.exact = exact
    p.suffix_min = suffix_min
    p.suffix_max = suffix_max
    return p


def _option_ok(
    p: _Problem, i: int, v: int, used: tuple[int, ...], next_used: list[int]
) -> bool:
    """前向检查：第 i 笔取值 v 后，各分组的剩余额度是否仍可凑出。"""
    for g in p.gm[i]:
        nu = used[g] + v
        pos = p.suffix_pos_after[g][i]
        target = p.t[g]
        if nu > target:
            return False
        remain_min = p.suffix_min[g][pos]
        remain_max = p.suffix_max[g][pos]
        if nu + remain_min > target or nu + remain_max < target:
            return False
        if p.exact[g] and (target - nu) not in p.reach_sets[g][pos]:
            return False
        next_used[g] = nu
    return True


def _group_reachable_sets(
    domains: Sequence[tuple[int, ...]],
    groups: Sequence[tuple[int, ...]],
    totals: Sequence[int],
    cap: int = 200_000,
) -> tuple[tuple[dict[int, frozenset[int]], ...], tuple[bool, ...]]:
    """对每个分组预计算“后缀笔数可达金额集合”。

    reach[k] 表示该分组中从第 k 个成员起（含）的笔数能凑出的金额集合。
    集合超过 cap 时停止精确计算，该分组退化为 min/max 兜底剪枝。
    """
    all_sets: list[tuple[dict[int, frozenset[int]], ...]] = []
    exact_flags: list[bool] = []
    for members, target in zip(groups, totals):
        n = len(members)
        reach: list[dict[int, frozenset[int]]] = [None] * (n + 1)  # type: ignore[list-item]
        reach[n] = {0: frozenset((0,))}
        exact = True
        for k in range(n - 1, -1, -1):
            prev = reach[k + 1]
            if not prev:
                # 后缀已无可达金额，本层同样为空，不能回退成 {0}
                reach[k] = {}
                continue
            merged: set[int] = set()
            for s in prev:
                for v in domains[members[k]]:
                    ns = s + v
                    if ns <= target:
                        merged.add(ns)
            if len(merged) > cap:
                exact = False
                reach[k] = {}
                break
            reach[k] = {s: frozenset((s,)) for s in merged}
        if not exact:
            reach = [{} for _ in range(n + 1)]
        all_sets.append(tuple(reach))
        exact_flags.append(exact)
    return tuple(all_sets), tuple(exact_flags)


def compute_feasible(
    domains: Sequence[Iterable[int]],
    group_memberships: Sequence[Sequence[int]],
    totals: Sequence[int],
    node_limit: int = 2_000_000,
    *,
    budget: SearchBudget | None = None,
) -> tuple[tuple[int, ...], ...]:
    """计算每笔票款的可行候选金额。

    参数
    ----
    domains:
        domains[i] 为第 i 笔票款的候选金额（整数分），去重后处理。
    group_memberships:
        group_memberships[i] 给出第 i 笔所属的各分组 id（本应用固定为
        [窗口分组 id, 售票日分组 id] 两套）。
    totals:
        各分组的纸面合计，下标为分组 id。
    node_limit:
        记忆化搜索访问状态数保护上限，超出抛 SearchLimitExceeded。
    budget:
        可选的共享搜索预算；传入时忽略 node_limit，状态数计入该预算。

    返回
    ----
    与 domains 等长的元组，第 i 项是该笔在至少一个完整可行解中出现过的
    取值集合（升序）。若整批无解，对应可行集合为空元组。
    """
    if budget is None:
        budget = SearchBudget(node_limit)
    if len(domains) == 0:
        return ()
    p = _prepare(domains, group_memberships, totals)
    n = p.n

    # dfs 成功时返回“子问题每笔可行取值”（长度 n-i）；无解返回 None
    memo: dict[tuple, tuple[tuple[int, ...], ...] | None] = {}

    def dfs(i: int, used: tuple[int, ...]):
        """返回从第 i 笔起的子问题中，每笔可行取值（子问题内下标 0..n-i-1）。"""
        budget.step()
        if i == n:
            # 终止：所有分组已用额度必须恰好等于纸面合计
            if all(used[g] == p.t[g] for g in range(p.group_count)):
                return ()
            return None

        key = (i, used)
        if key in memo:
            return memo[key]

        feasible: list[set[int]] = [set() for _ in range(n - i)]
        next_used = [0] * p.group_count
        for v in p.d[i]:
            for g in range(p.group_count):
                next_used[g] = used[g]
            if not _option_ok(p, i, v, used, next_used):
                continue

            sub = dfs(i + 1, tuple(next_used))
            if sub is None:
                continue
            # 该取值出现在至少一个完整可行解中
            feasible[0].add(v)
            for offset, vals in enumerate(sub, start=1):
                feasible[offset].update(vals)

        if not feasible[0]:
            result = None
        else:
            result = tuple(tuple(sorted(vals)) for vals in feasible)
        memo[key] = result
        return result

    root = dfs(0, tuple([0] * p.group_count))
    if root is None:
        return tuple(tuple() for _ in range(n))
    return root


def select_max_adoptions(
    domains: Sequence[Iterable[int]],
    group_memberships: Sequence[Sequence[int]],
    totals: Sequence[int],
    selections: Sequence[tuple[int, int]],
    node_limit: int = 2_000_000,
    *,
    budget: SearchBudget | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """在既有确认与两套合计下，挑出数量最多、能同时成立的勾选金额。

    selections 为 ``[(票款下标, 勾选金额), ...]``。返回
    ``(采纳, 未采纳)`` 两个列表，均按票款下标（清单行序）升序，元素原样
    保留传入的 ``(下标, 金额)``。

    搜索策略
    --------
    按清单行序对所有勾选做带记忆化的回溯：遇到勾选时先试“采纳”再试
    “不采纳”，其余笔自由取候选。以 (可行完整解, 采纳笔数, 取舍序列) 为
    状态缓存后缀最优；取舍序列按行序比较，1（采纳）优先。因此先比笔数、
    笔数相同再取第一笔取舍不同的方案。搜索状态数计入共享预算。
    """
    if budget is None:
        budget = SearchBudget(node_limit)
    p = _prepare(domains, group_memberships, totals)
    n = p.n

    # 勾选按行序排列；同一笔出现多次时由调用方保证不会发生
    ordered = sorted(selections, key=lambda item: item[0])
    selected: dict[int, int] = {}
    for i, v in ordered:
        if not 0 <= i < n:
            raise SolverError(f"勾选的票款下标 {i} 超出范围")
        if i in selected:
            raise SolverError(f"票款下标 {i} 被重复勾选")
        if v not in p.d[i]:
            raise SolverError(f"票款下标 {i} 的勾选金额 {v} 不在其候选金额中")
        selected[i] = v

    # 后缀最优：key=(i, 已用额度) -> (采纳笔数, 取舍元组) 或 None（无解）。
    # 取舍元组与位置 i..n-1 一一对应：1 表示采纳该行勾选，0 表示未勾选或
    # 不采纳。未勾选位固定为 0，不影响同笔数方案的先后比较。
    memo: dict[tuple, tuple[int, tuple[int, ...]] | None] = {}

    def dfs(i: int, used: tuple[int, ...]):
        budget.step()
        if i == n:
            if all(used[g] == p.t[g] for g in range(p.group_count)):
                return (0, ())
            return None

        key = (i, used)
        if key in memo:
            return memo[key]

        best: tuple[int, tuple[int, ...]] | None = None
        next_used = [0] * p.group_count

        def consider(choice: int, sub) -> None:
            nonlocal best
            if sub is None:
                return
            count, suffix = sub
            candidate = (count + choice, (choice,) + suffix)
            if best is None or candidate > best:
                best = candidate

        # 勾选笔：先试采纳（钉死为勾选金额），再试不采纳（候选域不变）
        pick = selected.get(i)
        if pick is not None:
            for g in range(p.group_count):
                next_used[g] = used[g]
            if _option_ok(p, i, pick, used, next_used):
                consider(1, dfs(i + 1, tuple(next_used)))
        for v in p.d[i]:
            if pick is not None and v == pick:
                # 不采纳却仍取同值，结果与采纳相同但少采纳一笔，必被支配
                continue
            for g in range(p.group_count):
                next_used[g] = used[g]
            if not _option_ok(p, i, v, used, next_used):
                continue
            consider(0, dfs(i + 1, tuple(next_used)))

        memo[key] = best
        return best

    root = dfs(0, tuple([0] * p.group_count))
    if root is None:
        raise SolverError("当前确认状态与两套纸面合计不存在可行解")
    _, indicator = root
    if len(indicator) != n:
        raise SolverError("搜索结果与票款行数不一致")
    adopted: list[tuple[int, int]] = []
    rejected: list[tuple[int, int]] = []
    for (i, v) in ordered:
        (adopted if indicator[i] else rejected).append((i, v))
    return adopted, rejected
