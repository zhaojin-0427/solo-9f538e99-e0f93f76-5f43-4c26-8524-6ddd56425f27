"""Flask 入仓校对应用。"""
from __future__ import annotations

import os

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from . import db as db_mod
from . import parsing, services
from .db import get_db, latest_batch_id, load_batch, load_batch_result, load_entries
from .db import load_groups, row_candidates, row_feasible
from .parsing import ParseError
from .solver import SearchLimitExceeded, SolverError
from .services import (
    AlreadyConfirmedError,
    InfeasibleError,
    SaveError,
    SelectionError,
    StaleBatchError,
)


def create_app(database: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DATABASE"] = database or os.environ.get(
        "RECON_DB_PATH", os.path.join(app.instance_path, "recon.sqlite3")
    )
    app.config["SECRET_KEY"] = os.environ.get("RECON_SECRET_KEY", "dev-secret-key")
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)

    app.teardown_appcontext(db_mod.close_db)

    with app.app_context():
        db_mod.init_db()

    def _current_batch_view():
        conn = get_db()
        batch_id = latest_batch_id(conn)
        if batch_id is None:
            return None
        batch = load_batch(conn, batch_id)
        entries = load_entries(conn, batch_id)
        groups = load_groups(conn, batch_id)
        view_entries = []
        for row in entries:
            view_entries.append(
                {
                    "id": row["id"],
                    "position": row["position"],
                    "ticket_no": row["ticket_no"],
                    "window": row["window_key"],
                    "sale_date": row["sale_date"],
                    "candidates": row_candidates(row),
                    "feasible": row_feasible(row),
                    "confirmed": row["confirmed_amount"],
                }
            )
        window_totals = [g for g in groups if g["group_type"] == "window"]
        date_totals = [g for g in groups if g["group_type"] == "date"]
        return {
            "batch": batch,
            "entries": view_entries,
            "window_totals": window_totals,
            "date_totals": date_totals,
            "all_singletons": services.all_singletons(conn, batch_id),
            "confirmed_count": sum(1 for e in view_entries if e["confirmed"] is not None),
            "last_result": load_batch_result(conn, batch_id),
        }

    @app.get("/")
    def index():
        view = _current_batch_view()
        if view is not None:
            try:
                view["search_nodes"] = services.count_search_nodes(
                    get_db(), view["batch"]["id"]
                )
            except (SolverError, InfeasibleError):
                # 展示用统计不可得时不影响页面其余内容
                view["search_nodes"] = None
        return render_template("index.html", view=view)

    @app.post("/batch/confirm")
    def confirm_batch():
        """批量提交多笔未确认票的勾选金额，一次重算。"""
        raw_batch = request.form.get("batch_id", "")
        try:
            batch_id = int(raw_batch)
        except (TypeError, ValueError):
            flash("提交失败：批次信息缺失，请刷新页面后重试。", "error")
            return redirect(url_for("index"))

        selections: list[services.BatchSelection] = []
        invalid: list[str] = []
        for raw_id, amount in request.form.items(multi=True):
            if not raw_id.startswith("amount_"):
                continue
            try:
                entry_id = int(raw_id[len("amount_"):])
                selections.append(services.BatchSelection(entry_id, int(amount)))
            except (TypeError, ValueError):
                invalid.append(raw_id)
        if invalid:
            flash("提交失败：勾选金额格式不正确，请刷新页面后重试。", "error")
            return redirect(url_for("index"))
        if not selections:
            flash("提交失败：请至少勾选一笔票款的候选金额。", "error")
            return redirect(url_for("index"))

        conn = get_db()
        try:
            result = services.confirm_entries(conn, batch_id, selections)
        except StaleBatchError:
            flash("提交失败：批次已切换，请刷新页面后重新校对。", "error")
            return redirect(url_for("index"))
        except AlreadyConfirmedError as exc:
            flash(f"提交失败：{exc}", "error")
            return redirect(url_for("index"))
        except SearchLimitExceeded as exc:
            flash(f"提交失败：{exc}。已保留原状，请减少勾选或缩小批次。", "error")
            return redirect(url_for("index"))
        except SaveError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except (SelectionError, InfeasibleError, SolverError) as exc:
            flash(f"提交失败，已保留原状：{exc}", "error")
            return redirect(url_for("index"))

        if result.rejected:
            flash(
                f"已采纳 {result.adopted_count} 笔确认，"
                f"{result.rejected_count} 笔未能同时成立，已保留原候选。",
                "ok",
            )
        else:
            flash(f"已采纳全部 {result.adopted_count} 笔勾选并重算可行金额。", "ok")
        return redirect(url_for("index"))

    @app.post("/upload")
    def upload():
        files = {
            "entries": request.files.get("entries_file"),
            "window": request.files.get("window_file"),
            "date": request.files.get("date_file"),
        }
        missing = [name for name, f in files.items() if f is None or f.filename == ""]
        if missing:
            labels = {"entries": "票款清单", "window": "窗口合计表", "date": "售票日合计表"}
            flash("请同时选择三个 CSV 文件，缺少：" + "、".join(labels[m] for m in missing), "error")
            return redirect(url_for("index"))
        try:
            parsed = parsing.parse_batch(
                files["entries"].read(),
                files["window"].read(),
                files["date"].read(),
            )
            conn = get_db()
            batch_id = services.create_batch(conn, parsed)
        except (ParseError, InfeasibleError, SolverError, SearchLimitExceeded) as exc:
            flash(f"上传失败：{exc}", "error")
            return redirect(url_for("index"))
        flash(f"第 {batch_id} 批已入库，共 {len(parsed.entries)} 笔，已算出各笔可行金额。", "ok")
        return redirect(url_for("index"))

    @app.post("/entries/<int:entry_id>/confirm")
    def confirm(entry_id: int):
        raw = request.form.get(f"amount_{entry_id}", request.form.get("amount", ""))
        try:
            amount = int(raw)
        except (TypeError, ValueError):
            flash("提交失败：请选择一个候选金额", "error")
            return redirect(url_for("index"))
        conn = get_db()
        batch_id = latest_batch_id(conn)
        if batch_id is None:
            flash("提交失败：当前没有批次", "error")
            return redirect(url_for("index"))
        try:
            services.confirm_entry(conn, batch_id, entry_id, amount)
        except (InfeasibleError, SolverError, SearchLimitExceeded) as exc:
            flash(f"确认冲突，已保留原有确认记录：{exc}", "error")
            return redirect(url_for("index"))
        ticket_no = request.form.get("ticket_no", "")
        if not ticket_no:
            row = conn.execute(
                "SELECT ticket_no FROM entries WHERE id = ?", (entry_id,)
            ).fetchone()
            ticket_no = row["ticket_no"] if row else ""
        flash(f"票款 {ticket_no} 已确认 {amount} 分，剩余候选已更新。", "ok")
        return redirect(url_for("index"))

    @app.post("/entries/<int:entry_id>/revoke")
    def revoke(entry_id: int):
        conn = get_db()
        batch_id = latest_batch_id(conn)
        if batch_id is None:
            flash("操作失败：当前没有批次", "error")
            return redirect(url_for("index"))
        try:
            services.revoke_entry(conn, batch_id, entry_id)
        except (InfeasibleError, SolverError, SearchLimitExceeded) as exc:
            flash(f"撤销失败：{exc}", "error")
            return redirect(url_for("index"))
        flash("已撤销该笔确认并重新计算可行金额。", "ok")
        return redirect(url_for("index"))

    @app.get("/export")
    def export():
        conn = get_db()
        batch_id = latest_batch_id(conn)
        if batch_id is None:
            flash("当前没有批次，无法下载", "error")
            return redirect(url_for("index"))
        try:
            content = services.export_csv(conn, batch_id)
        except InfeasibleError as exc:
            flash(f"暂不能入仓：{exc}", "error")
            return redirect(url_for("index"))
        filename = f"warehouse_batch_{batch_id}.csv"
        return Response(
            content,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    return app
