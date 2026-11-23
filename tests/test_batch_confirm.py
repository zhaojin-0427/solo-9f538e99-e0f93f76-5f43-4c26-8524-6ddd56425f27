"""批量勾选确认：一次提交、最多采纳、行序取舍、预算与回滚。"""
import io

import pytest

from app import create_app, solver, services

ENTRIES = """票号,窗口,售票日,候选金额1,候选金额2,候选金额3
T001,W1,2026-11-01,1000,1200,1500
T002,W1,2026-11-01,800,900
T003,W1,2026-11-02,2200,2300,2500
T004,W2,2026-11-01,500,600,700
T005,W2,2026-11-02,3000,3100,3200
T006,W2,2026-11-02,400,450,500
"""
WINDOW_TOTALS = "窗口,合计\nW1,4200\nW2,4150\n"
DATE_TOTALS = "售票日,合计\n2026-11-01,2500\n2026-11-02,5850\n"


@pytest.fixture()
def app(tmp_path):
    app = create_app(database=str(tmp_path / "test.sqlite3"))
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def upload_batch(client, entries=ENTRIES, window_totals=WINDOW_TOTALS,
                 date_totals=DATE_TOTALS):
    def data(text):
        return (io.BytesIO(text.encode("utf-8")), "f.csv")

    return client.post(
        "/upload",
        data={
            "entries_file": data(entries),
            "window_file": data(window_totals),
            "date_file": data(date_totals),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def entry_ids(app):
    from app.db import get_db, latest_batch_id, load_entries

    with app.app_context():
        conn = get_db()
        rid = latest_batch_id(conn)
        rows = load_entries(conn, rid)
    return {r["ticket_no"]: r["id"] for r in rows}, rid


def latest_batch(app):
    with app.app_context():
        from app.db import get_db, latest_batch_id

        conn = get_db()
        return latest_batch_id(conn)


def post_batch(client, batch_id, pairs, **extra):
    from werkzeug.datastructures import MultiDict

    data = MultiDict([("batch_id", str(batch_id))])
    for entry_id, amount in pairs:
        data.add(f"amount_{entry_id}", str(amount))
    for k, v in extra.items():
        data.add(k, str(v))
    return client.post("/batch/confirm", data=data, follow_redirects=True)


def rows_by_ticket(app):
    from app.db import get_db, latest_batch_id, load_entries, row_feasible

    with app.app_context():
        conn = get_db()
        rows = load_entries(conn, latest_batch_id(conn))
        return {
            r["ticket_no"]: {
                "confirmed": r["confirmed_amount"],
                "feasible": row_feasible(r),
                "candidates": __import__("json").loads(r["candidates"]),
            }
            for r in rows
        }


def test_batch_accepts_maximum_and_reports_rejected(client, app):
    upload_batch(client)
    ids, batch_id = entry_ids(app)

    # 勾选解 A 的 4 笔 + 解 B 的 T005/T006；T005=3200 与解 A 互斥，
    # 但 T006=450 在两个解里都成立，故最多同时采纳 5 笔。
    resp = post_batch(
        client,
        batch_id,
        [
            (ids["T001"], 1000),
            (ids["T002"], 900),
            (ids["T003"], 2300),
            (ids["T004"], 600),
            (ids["T005"], 3200),
            (ids["T006"], 450),
        ],
    )
    body = resp.data.decode()
    assert "已采纳 5 笔确认" in body
    assert "1 笔未能同时成立" in body
    assert "未采纳（请翻纸复查）" in body
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] == 1000
    assert state["T002"]["confirmed"] == 900
    assert state["T006"]["confirmed"] == 450
    assert state["T005"]["confirmed"] is None
    assert state["T005"]["candidates"] == [3000, 3100, 3200]
    # 采纳后其他笔已被钉住
    assert state["T005"]["feasible"] == (3100,)
    # 页面列出未采纳票号与本次访问的搜索状态数
    assert "T005" in body
    assert "本次访问的搜索状态数" in body


def test_batch_tie_breaks_by_row_order(client, app):
    upload_batch(client)
    ids, batch_id = entry_ids(app)
    # T001=1000 属解 A，T002=800 属解 B，两笔互斥；同样只能采纳 1 笔时，
    # 清单行序靠前的 T001 优先。
    resp = post_batch(client, batch_id, [(ids["T002"], 800), (ids["T001"], 1000)])
    body = resp.data.decode()
    assert "已采纳 1 笔确认" in body
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] == 1000
    assert state["T002"]["confirmed"] is None
    # 顺序无关：提交顺序颠倒，仍按清单行序采纳 T001
    client.post("/entries/%d/revoke" % ids["T001"], follow_redirects=True)
    resp = post_batch(client, batch_id, [(ids["T001"], 1000), (ids["T002"], 800)])
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] == 1000
    assert state["T002"]["confirmed"] is None


def test_batch_compatible_picks_all_adopted(client, app):
    """相互兼容的勾选应全部采纳（T002=800 与 T001=1200 同属解 B）。"""
    upload_batch(client)
    ids, batch_id = entry_ids(app)
    resp = post_batch(client, batch_id, [(ids["T002"], 800), (ids["T001"], 1200)])
    body = resp.data.decode()
    assert "已采纳全部 2 笔" in body
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] == 1200
    assert state["T002"]["confirmed"] == 800


def test_batch_tie_same_count_prefers_earlier_difference(client, app):
    """两笔勾选互斥时，采纳笔数相同则比较第一笔不同取舍。"""
    upload_batch(client)
    ids, batch_id = entry_ids(app)
    # T004=500 与 T001=1000 属于解 B 与解 A，不能同时成立；
    # 分别单独勾选都能成立。两笔都勾且互斥时按行序优先 T001。
    resp = post_batch(
        client, batch_id, [(ids["T004"], 500), (ids["T001"], 1000)]
    )
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] == 1000
    assert state["T004"]["confirmed"] is None


def test_batch_rejects_when_selected_already_confirmed(client, app):
    upload_batch(client)
    ids, batch_id = entry_ids(app)
    post_batch(client, batch_id, [(ids["T001"], 1000)])
    resp = post_batch(
        client, batch_id, [(ids["T001"], 1000), (ids["T002"], 900)]
    )
    assert "已有确认" in resp.data.decode()
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] == 1000
    assert state["T002"]["confirmed"] is None


def test_batch_rejects_stale_batch(client, app):
    upload_batch(client)
    ids, old_batch = entry_ids(app)
    upload_batch(client)  # 换新批
    assert latest_batch(app) != old_batch
    resp = post_batch(client, old_batch, [(ids["T001"], 1000)])
    assert "批次已切换" in resp.data.decode()
    state = rows_by_ticket(app)
    assert all(v["confirmed"] is None for v in state.values())


def test_batch_budget_exhausted_keeps_state(client, app, monkeypatch):
    upload_batch(client)
    ids, batch_id = entry_ids(app)

    def boom(*args, **kwargs):
        raise solver.SearchLimitExceeded("搜索状态数超过上限（5）")

    monkeypatch.setattr(services.csp, "select_max_adoptions", boom)
    resp = post_batch(
        client, batch_id, [(ids["T001"], 1000), (ids["T002"], 900)]
    )
    body = resp.data.decode()
    assert "预算已用尽" in body or "搜索状态数超过上限" in body
    state = rows_by_ticket(app)
    assert all(v["confirmed"] is None for v in state.values())
    assert state["T001"]["feasible"] == (1000, 1200)


def test_batch_shared_budget_counts_both_searches(app, client, monkeypatch):
    """采纳判定和全表重算共用一个预算对象，合计状态数不超过上限。"""
    upload_batch(client)
    ids, batch_id = entry_ids(app)
    seen = {}
    real_select = solver.select_max_adoptions
    real_feasible = solver.compute_feasible

    def spy_select(*a, **kw):
        seen["select_budget"] = kw.get("budget")
        return real_select(*a, **kw)

    def spy_feasible(*a, **kw):
        seen.setdefault("feasible_budget", kw.get("budget"))
        return real_feasible(*a, **kw)

    monkeypatch.setattr(services.csp, "select_max_adoptions", spy_select)
    monkeypatch.setattr(services.csp, "compute_feasible", spy_feasible)
    with app.app_context():
        from app.db import get_db

        conn = get_db()
        result = services.confirm_entries(
            conn,
            batch_id,
            [services.BatchSelection(ids["T001"], 1000)],
            node_limit=50,
        )
    assert seen["select_budget"] is seen["feasible_budget"]
    assert seen["select_budget"].used == result.search_nodes
    assert result.search_nodes <= 50


def _break_saving(monkeypatch):
    import sqlite3

    class _FailingJson:
        @staticmethod
        def dumps(*args, **kwargs):
            raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(services, "json", _FailingJson)


def test_batch_save_failure_rolls_back(client, app, monkeypatch):
    upload_batch(client)
    ids, batch_id = entry_ids(app)

    _break_saving(monkeypatch)
    with app.app_context():
        from app.db import get_db

        conn = get_db()
        with pytest.raises(services.SaveError):
            services.confirm_entries(
                conn,
                batch_id,
                [services.BatchSelection(ids["T001"], 1000)],
            )
    monkeypatch.undo()

    # 回滚后原有状态不变：无确认、无确认流水
    state = rows_by_ticket(app)
    assert all(v["confirmed"] is None for v in state.values())
    with app.app_context():
        from app.db import get_db

        conn = get_db()
        assert conn.execute(
            "SELECT COUNT(*) c FROM confirmations"
        ).fetchone()["c"] == 0


def test_batch_route_reports_save_failure(client, app, monkeypatch):
    upload_batch(client)
    ids, batch_id = entry_ids(app)

    _break_saving(monkeypatch)
    resp = post_batch(client, batch_id, [(ids["T001"], 1000)])
    monkeypatch.undo()
    body = resp.data.decode()
    assert "保存失败" in body
    state = rows_by_ticket(app)
    assert state["T001"]["confirmed"] is None


def test_batch_rejects_unknown_or_repeated_selection(client, app):
    upload_batch(client)
    ids, batch_id = entry_ids(app)
    resp = post_batch(client, batch_id, [(ids["T001"], 999)])
    assert "不在该笔票款的候选金额中" in resp.data.decode()
    resp = post_batch(client, batch_id, [(ids["T001"], 1000), (ids["T001"], 1200)])
    assert "只能勾选一个金额" in resp.data.decode()
    assert all(v["confirmed"] is None for v in rows_by_ticket(app).values())


def test_batch_empty_selection_rejected(client, app):
    upload_batch(client)
    _, batch_id = entry_ids(app)
    resp = post_batch(client, batch_id, [])
    assert "至少勾选一笔" in resp.data.decode()


def test_no_full_solution_rejected(client, app, monkeypatch):
    """无完整解时不做任何改动。"""
    upload_batch(client)
    ids, batch_id = entry_ids(app)

    def no_solution(*a, **kw):
        raise solver.SolverError("当前确认状态与两套纸面合计不存在可行解")

    monkeypatch.setattr(services.csp, "select_max_adoptions", no_solution)
    resp = post_batch(client, batch_id, [(ids["T001"], 1000)])
    assert "不存在可行解" in resp.data.decode()
    assert all(v["confirmed"] is None for v in rows_by_ticket(app).values())
