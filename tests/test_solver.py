import pytest

from app.solver import (
    SearchBudget,
    SearchLimitExceeded,
    SolverError,
    compute_feasible,
    select_max_adoptions,
)


def test_single_solution_exact_domains():
    # 每个分组只有一种凑法时，所有可行域都收缩到唯一值
    # 唯一完整解：(100, 400, 600, 150)
    domains = [
        (100, 150),
        (350, 400),
        (550, 600),
        (100, 150),
    ]
    # 四个分组：g0={0,1}=500, g1={2,3}=750, g2={0,2}=700, g3={1,3}=550
    memberships = [(0, 2), (0, 3), (1, 2), (1, 3)]
    totals = (500, 750, 700, 550)
    result = compute_feasible(domains, memberships, totals)
    assert result == ((100,), (400,), (600,), (150,))


def test_multiple_solutions_keeps_values_appearing_in_any_solution():
    # 4 笔四个分组；两套合计
    # g0: x0+x1=10, g1: x2+x3=20；g2: x0+x2=15, g3: x1+x3=15
    domains = [(5, 8), (2, 5), (7, 10), (10, 13)]
    memberships = [(0, 2), (0, 3), (1, 2), (1, 3)]
    result = compute_feasible(domains, memberships, (10, 20, 15, 15))
    # 可行完整解：(5,5,10,10) 与 (8,2,7,13)
    assert result == (
        (5, 8),
        (2, 5),
        (7, 10),
        (10, 13),
    )


def test_infeasible_returns_empty_domains():
    domains = [(1, 2), (3, 4)]
    memberships = [(0, 1), (0, 1)]
    # 唯一分组合计 100，无法凑出
    result = compute_feasible(domains, memberships, (100, 100))
    assert result == (tuple(), tuple())


def test_fixed_confirmation_shrinks_other_domains():
    # 把某笔域收缩成单值，模拟复核员确认后的重算
    memberships = [(0, 2), (0, 3), (1, 2), (1, 3)]
    result = compute_feasible(
        [(8,), (2, 5), (7, 10), (10, 13)],
        memberships,
        (10, 20, 15, 15),
    )
    # 确认 x0=8 后只剩 (8,2,7,13)
    assert result == ((8,), (2,), (7,), (13,))


def test_conflicting_confirmation_yields_empty():
    domains = [(8,), (2, 5), (7, 10), (10, 13)]
    memberships = [(0, 2), (0, 3), (1, 2), (1, 3)]
    result = compute_feasible(domains, memberships, (9, 20, 15, 15))
    assert any(len(v) == 0 for v in result)


def test_amount_bounds_large_values_are_exact():
    # 金额上限 1_000_000，求解结果必须精确
    big = 1_000_000
    domains = [
        (big - 1, big),
        (1, 2),
        (big - 3, big - 2),
        (3, 4),
    ]
    memberships = [(0, 2), (0, 3), (1, 2), (1, 3)]
    totals = (big + 1, big, big + big - 3, 4)
    result = compute_feasible(domains, memberships, totals)
    # g0: x0+x1=1000001 -> x0=1000000,x1=1；g3: x1+x3=4 -> x3=3
    # g2: x0+x2=1999997 -> x2=999997；g1: x2+x3=1000000 校验通过
    assert result == ((big,), (1,), (big - 3,), (3,))


def test_validation_errors():
    with pytest.raises(SolverError):
        compute_feasible([()], [(0,)], (1,))
    with pytest.raises(SolverError):
        compute_feasible([(1,)], [()], ())
    with pytest.raises(SolverError):
        compute_feasible([(1,)], [(0, 1)], (1,))  # totals 数量不符


# 两个解：A=(5,5,10,10)，B=(8,2,7,13)
MULTI_DOMAINS = [(5, 8), (2, 5), (7, 10), (10, 13)]
MULTI_MEMBERSHIPS = [(0, 2), (0, 3), (1, 2), (1, 3)]
MULTI_TOTALS = (10, 20, 15, 15)


def test_select_max_adoptions_all_compatible():
    adopted, rejected = select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS,
        [(0, 5), (1, 5), (2, 10), (3, 10)],
    )
    assert adopted == [(0, 5), (1, 5), (2, 10), (3, 10)]
    assert rejected == []


def test_select_max_adoptions_prefers_more_adoptions():
    # 解 A=(5,5,10,10)，解 B=(8,2,7,13)。
    # 勾选 (0,8) 只在解 B 成立；(1,5),(2,10),(3,10) 只在解 A 成立。
    # 采纳 (0,8) 至多 1 笔，采纳解 A 三笔则可同时成立 3 笔，故取 3 笔。
    adopted, rejected = select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS,
        [(3, 10), (2, 10), (1, 5), (0, 8)],
    )
    assert adopted == [(1, 5), (2, 10), (3, 10)]
    assert rejected == [(0, 8)]


def test_select_max_adoptions_tie_break_by_row_order():
    # (0,8) 只在解 B 成立，(2,10) 只在解 A 成立，两笔互斥，均只能采纳 1 笔；
    # 笔数相同按清单行序比较，第 0 笔的取舍优先。
    adopted, rejected = select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, [(2, 10), (0, 8)]
    )
    assert adopted == [(0, 8)]
    assert rejected == [(2, 10)]


def test_select_max_adoptions_same_solution_all_adopted():
    # 解 B 为 (8,2,7,13)，(0,8) 与 (1,2) 可同时采纳。
    adopted, rejected = select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, [(0, 8), (1, 2)]
    )
    assert adopted == [(0, 8), (1, 2)]
    assert rejected == []


def test_select_max_adoptions_indifferent_values_do_not_block():
    # 勾选与另一解相同的值但非该笔的取值：解 A 中 (2,10) 不在候选？10 在候选中。
    adopted, rejected = select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, [(2, 10)]
    )
    assert adopted == [(2, 10)]


def test_select_max_adoptions_shared_budget_and_limit():
    budget = SearchBudget(limit=5)
    with pytest.raises(SearchLimitExceeded):
        select_max_adoptions(
            MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS,
            [(0, 5)], budget=budget,
        )
    assert budget.used == 6


def test_select_max_adoptions_rejects_bad_input():
    with pytest.raises(SolverError, match="不在其候选"):
        select_max_adoptions(
            MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, [(0, 999)]
        )
    with pytest.raises(SolverError, match="重复勾选"):
        select_max_adoptions(
            MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS,
            [(0, 5), (0, 8)],
        )
    with pytest.raises(SolverError, match="不存在可行解"):
        select_max_adoptions(
            [(1,), (2,)], [(0, 1), (0, 1)], (100, 100), []
        )


def test_select_max_adoptions_no_selections_returns_empty():
    adopted, rejected = select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, []
    )
    assert adopted == [] and rejected == []


def test_budget_shared_between_searches():
    budget = SearchBudget()
    select_max_adoptions(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, [(0, 5)], budget=budget
    )
    used_after_select = budget.used
    compute_feasible(
        MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, budget=budget
    )
    assert budget.used > used_after_select


def test_node_limit_still_applies_without_budget():
    with pytest.raises(SearchLimitExceeded):
        compute_feasible(
            MULTI_DOMAINS, MULTI_MEMBERSHIPS, MULTI_TOTALS, node_limit=1
        )
