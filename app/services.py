"""业务编排：把持久化数据喂给求解器，落库确认结果，生成入仓 CSV。"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass

from . import solver as csp
from .db import (
    load_entries,
    load_groups,
    row_candidates,
    row_feasible,
    save_batch_result,
)
from .parsing import ParsedBatch

DEFAULT_NODE_LIMIT = 2_000_000


class InfeasibleError(Exception):
    """当前确认状态与两套合计无共同可行解。"""


class StaleBatchError(Exception):
    """提交所针对的批次已不是当前批次（页面过期）。"""


class AlreadyConfirmedError(Exception):
    """所勾选的票款在提交时已有确认（页面过期）。"""


class SelectionError(ValueError):
    """勾选内容不合法（重复勾选、金额不在候选中等）。"""


class SaveError(Exception):
    """写入数据库失败，已回滚，原有状态保持不变。"""


def _problem_parts(conn: sqlite3.Connection, batch_id: int):
    entry_rows = load_entries(conn, batch_id)
    group_rows = load_groups(conn, batch_id)

    window_keys = [r["group_key"] for r in group_rows if r["group_type"] == "window"]
    date_keys = [r["group_key"] for r in group_rows if r["group_type"] == "date"]
    window_index = {k: i for i, k in enumerate(window_keys)}
    date_offset = len(window_keys)
    date_index = {k: i + date_offset for i, k in enumerate(date_keys)}
    totals = [r["total"] for r in group_rows if r["group_type"] == "window"]
    totals += [r["total"] for r in group_rows if r["group_type"] == "date"]

    domains: list[tuple[int, ...]] = []
    memberships: list[tuple[int, int]] = []
    for row in entry_rows:
        confirmed = row["confirmed_amount"]
        if confirmed is None:
            domains.append(row_candidates(row))
        else:
            domains.append((confirmed,))
        memberships.append(
            (window_index[row["window_key"]], date_index[row["sale_date"]])
        )
    return domains, memberships, totals, entry_rows


def build_problem(conn: sqlite3.Connection, batch_id: int):
    """从库里组装求解器输入。

    已确认金额的笔，候选域收缩为单值（确认值）；未确认笔保留原候选。
    返回 (domains, memberships, totals, entry_rows)，其中分组 id 布局为
    前半段窗口分组、后半段售票日分组。
    """
    return _problem_parts(conn, batch_id)


def _budget(node_limit: int) -> csp.SearchBudget:
    return csp.SearchBudget(node_limit)

def recompute_feasible(
    conn: sqlite3.Connection,
    batch_id: int,
    *,
    persist: bool = True,
    budget: csp.SearchBudget | None = None,
) -> list[tuple[int, ...]]:
    """按当前确认状态重算每笔可行金额，可选择写回 entries.feasible。"""
    domains, memberships, totals, entry_rows = build_problem(conn, batch_id)
    feasible = csp.compute_feasible(
        domains,
        memberships,
        totals,
        budget=budget if budget is not None else _budget(DEFAULT_NODE_LIMIT),
    )
    if not feasible or any(len(v) == 0 for v in feasible):
        raise InfeasibleError("当前票款候选与两套纸面合计不存在可行解")
    if persist:
        for row, vals in zip(entry_rows, feasible):
            conn.execute(
                "UPDATE entries SET feasible = ? WHERE id = ?",
                (json.dumps(list(vals)), row["id"]),
            )
    return [tuple(v) for v in feasible]


def create_batch(conn: sqlite3.Connection, parsed: ParsedBatch) -> int:
    """整批入库并完成首次可行性计算；无解则抛 InfeasibleError 并不留数据。"""
    cur = conn.execute(
        "INSERT INTO batches (entry_count) VALUES (?)", (len(parsed.entries),)
    )
    batch_id = cur.lastrowid

    window_total_rows = {r.group_key: r.total for r in parsed.window_totals}
    date_total_rows = {r.group_key: r.total for r in parsed.date_totals}
    for key, total in window_total_rows.items():
        conn.execute(
            "INSERT INTO groups (batch_id, group_type, group_key, total) "
            "VALUES (?, 'window', ?, ?)",
            (batch_id, key, total),
        )
    for key, total in date_total_rows.items():
        conn.execute(
            "INSERT INTO groups (batch_id, group_type, group_key, total) "
            "VALUES (?, 'date', ?, ?)",
            (batch_id, key, total),
        )
    for pos, e in enumerate(parsed.entries):
        conn.execute(
            "INSERT INTO entries "
            "(batch_id, position, ticket_no, window_key, sale_date, "
            " candidates, confirmed_amount, feasible) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, '[]')",
            (
                batch_id,
                pos,
                e.ticket_no,
                e.window,
                e.sale_date,
                json.dumps(list(e.candidates)),
            ),
        )
    try:
        recompute_feasible(conn, batch_id)
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return batch_id


def confirm_entry(
    conn: sqlite3.Connection, batch_id: int, entry_id: int, amount: int
) -> None:
    """确认单笔金额：与最新已确认金额一起必须仍有解，否则全部回滚。"""
    row = conn.execute(
        "SELECT * FROM entries WHERE id = ? AND batch_id = ?",
        (entry_id, batch_id),
    ).fetchone()
    if row is None:
        raise InfeasibleError("票款记录不存在或不属于当前批次")
    candidates = row_candidates(row)
    if amount not in candidates:
        raise InfeasibleError("所选金额不在该笔票款的候选金额中")

    try:
        conn.execute(
            "UPDATE entries SET confirmed_amount = ? WHERE id = ?",
            (amount, entry_id),
        )
        recompute_feasible(conn, batch_id)
    except Exception:
        conn.rollback()
        raise
    conn.execute(
        "INSERT INTO confirmations (batch_id, entry_id, amount, action) "
        "VALUES (?, ?, ?, 'confirm')",
        (batch_id, entry_id, amount),
    )
    conn.commit()


@dataclass(frozen=True)
class BatchSelection:
    """一笔待批量确认的勾选：entry_id 与所选金额（分）。"""

    entry_id: int
    amount: int


@dataclass(frozen=True)
class BatchConfirmationResult:
    """批量确认的结果：采纳/未采纳的票号与本次搜索访问的状态数。"""

    adopted: tuple[tuple[str, int], ...]
    rejected: tuple[tuple[str, int], ...]
    search_nodes: int

    @property
    def adopted_count(self) -> int:
        return len(self.adopted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def confirm_entries(
    conn: sqlite3.Connection,
    batch_id: int,
    selections: list[BatchSelection],
    *,
    node_limit: int = DEFAULT_NODE_LIMIT,
) -> BatchConfirmationResult:
    """批量确认：一次提交里采纳最多能同时成立的勾选。

    - 只在当前批次上操作；提交时批次已切换则拒绝（StaleBatchError）。
    - 所选票款在提交时已有确认则拒绝（AlreadyConfirmedError），不做部分采纳。
    - 采纳方案：先比采纳笔数，笔数相同按清单行序比较，优先采纳第一笔取舍
      不同的票（即更靠前的票优先采纳）。
    - 采纳判定与随后的一次全表可行金额重算共用同一搜索预算。
    - 采纳的确认、确认流水与新的 feasible 在同一事务内保存；任何失败都回滚。
    - 未采纳的票保持原状，候选金额照旧，供后续校对。
    """
    latest = conn.execute("SELECT MAX(id) AS id FROM batches").fetchone()["id"]
    if latest is None or latest != batch_id:
        raise StaleBatchError("该批次已不是当前批次，请刷新页面后重试")

    entries = load_entries(conn, batch_id)
    if not entries:
        raise InfeasibleError("当前批次没有任何票款记录")
    by_id = {row["id"]: pos for pos, row in enumerate(entries)}

    # 勾选校验：票款存在、金额在候选中、同一笔不重复、尚未确认
    chosen: dict[int, int] = {}
    for sel in selections:
        pos = by_id.get(sel.entry_id)
        if pos is None:
            raise SelectionError("所选票款不存在或不属于当前批次，请刷新页面")
        if pos in chosen:
            raise SelectionError("同一笔票款只能勾选一个金额")
        if entries[pos]["confirmed_amount"] is not None:
            raise AlreadyConfirmedError(
                "所选票款已有确认，请刷新页面后重新校对"
            )
        if sel.amount not in row_candidates(entries[pos]):
            raise SelectionError("所选金额不在该笔票款的候选金额中")
        chosen[pos] = sel.amount

    domains, memberships, totals, entry_rows = _problem_parts(conn, batch_id)
    budget = _budget(node_limit)
    picks = sorted(chosen.items())
    try:
        adopted, rejected = csp.select_max_adoptions(
            domains, memberships, totals, picks, budget=budget
        )
        if adopted:
            for pos, amount in adopted:
                domains[pos] = (amount,)
        feasible = csp.compute_feasible(
            domains, memberships, totals, budget=budget
        )
    except csp.SearchLimitExceeded as exc:
        raise csp.SearchLimitExceeded(
            f"{exc}；本次预算已用尽，未做任何改动，请减少勾选或缩小批次"
        ) from exc
    nodes = budget.used

    if not feasible or any(len(v) == 0 for v in feasible):
        raise InfeasibleError("采纳后与两套纸面合计不存在可行解")

    adopted_labeled = tuple(
        (entry_rows[pos]["ticket_no"], amount) for pos, amount in adopted
    )
    rejected_labeled = tuple(
        (entry_rows[pos]["ticket_no"], amount) for pos, amount in rejected
    )
    try:
        for pos, amount in adopted:
            entry_id = entry_rows[pos]["id"]
            conn.execute(
                "UPDATE entries SET confirmed_amount = ? WHERE id = ?",
                (amount, entry_id),
            )
            conn.execute(
                "INSERT INTO confirmations (batch_id, entry_id, amount, action) "
                "VALUES (?, ?, ?, 'confirm')",
                (batch_id, entry_id, amount),
            )
        for row, vals in zip(entry_rows, feasible):
            conn.execute(
                "UPDATE entries SET feasible = ? WHERE id = ?",
                (json.dumps(list(vals)), row["id"]),
            )
        save_batch_result(
            conn,
            batch_id,
            {
                "adopted": [list(item) for item in adopted_labeled],
                "rejected": [list(item) for item in rejected_labeled],
                "search_nodes": nodes,
            },
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise SaveError(f"保存失败，已保留原状：{exc}") from exc

    return BatchConfirmationResult(
        adopted=adopted_labeled,
        rejected=rejected_labeled,
        search_nodes=nodes,
    )


def count_search_nodes(
    conn: sqlite3.Connection,
    batch_id: int,
    *,
    node_limit: int = DEFAULT_NODE_LIMIT,
) -> int:
    """只读地跑一次全表可行金额搜索，返回访问的状态数（供页面展示）。"""
    domains, memberships, totals, _ = build_problem(conn, batch_id)
    budget = _budget(node_limit)
    csp.compute_feasible(domains, memberships, totals, budget=budget)
    return budget.used


def revoke_entry(conn: sqlite3.Connection, batch_id: int, entry_id: int) -> None:
    """撤销单笔确认，恢复其候选域并重算其余可行金额。"""
    row = conn.execute(
        "SELECT id, confirmed_amount FROM entries WHERE id = ? AND batch_id = ?",
        (entry_id, batch_id),
    ).fetchone()
    if row is None:
        raise InfeasibleError("票款记录不存在或不属于当前批次")
    if row["confirmed_amount"] is None:
        return
    conn.execute(
        "UPDATE entries SET confirmed_amount = NULL WHERE id = ?", (entry_id,)
    )
    conn.execute(
        "INSERT INTO confirmations (batch_id, entry_id, amount, action) "
        "VALUES (?, ?, ?, 'revoke')",
        (batch_id, entry_id, row["confirmed_amount"]),
    )
    recompute_feasible(conn, batch_id)
    conn.commit()


def all_singletons(conn: sqlite3.Connection, batch_id: int) -> bool:
    """每笔只剩一个可行金额（含已确认笔）时才允许下载。"""
    rows = load_entries(conn, batch_id)
    return bool(rows) and all(len(row_feasible(r)) == 1 for r in rows)


def export_csv(conn: sqlite3.Connection, batch_id: int) -> str:
    """生成入仓 CSV：原票号、窗口、售票日、确定金额（单位：分）。"""
    # 导出前再实时求解一次，防止库内 feasible 过期
    feasible = recompute_feasible(conn, batch_id, persist=False)
    if any(len(v) != 1 for v in feasible):
        raise InfeasibleError("仍有票款存在多个可行金额，暂不能入仓")
    rows = load_entries(conn, batch_id)
    buf = io.StringIO()
    buf.write("﻿")
    writer = csv.writer(buf)
    writer.writerow(["票号", "窗口", "售票日", "金额"])
    for row, vals in zip(rows, feasible):
        writer.writerow(
            [row["ticket_no"], row["window_key"], row["sale_date"], vals[0]]
        )
    return buf.getvalue()
