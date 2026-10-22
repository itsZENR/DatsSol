"""FastAPI server exposing Go-dashboard API shape.

Backed by CSV dump for historical rounds + optional live arena poller.

Usage:
    python -m simulator.dashboard.server --dump dump --port 8080        # only CSV
    python -m simulator.dashboard.server --live --port 8080             # +live
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from simulator.dashboard.csv_store import CsvStore
from simulator.dashboard.live_store import LiveStore, LIVE_ROUND_ID


def build_app(store: CsvStore, live: Optional[LiveStore], static_dir: Path) -> FastAPI:
    app = FastAPI(title="DatsSol Dashboard")

    def _is_live(round_id: int) -> bool:
        return live is not None and round_id == LIVE_ROUND_ID

    @app.get("/api/rounds")
    def rounds():
        out = store.get_rounds()
        if live is not None:
            summary = live.get_round_summary()
            if summary is not None:
                out.append(summary)
        return out

    @app.get("/api/scores")
    def scores(round_id: int = Query(1)):
        if _is_live(round_id):
            return live.get_score_history()
        return store.get_score_history(round_id)

    @app.get("/api/map")
    def map_(round_id: int = Query(1)):
        if _is_live(round_id):
            return live.get_mountains()
        return store.get_mountains(round_id)

    @app.get("/api/turns")
    def turns(round_id: int = Query(1)):
        if _is_live(round_id):
            return live.get_turn_list()
        return store.get_turn_list(round_id)

    @app.get("/api/turn")
    def turn(round_id: int = Query(1), turn_no: int = Query(1)):
        if _is_live(round_id):
            return JSONResponse(live.get_turn_data(turn_no))
        return JSONResponse(store.get_turn_data(round_id, turn_no))

    @app.get("/api/round_stats")
    def round_stats(round_id: int = Query(1)):
        if _is_live(round_id):
            return live.get_round_stats()
        return store.get_round_stats(round_id)

    @app.get("/api/accumulated")
    def accumulated(round_id: int = Query(1), turn_no: int = Query(1)):
        if _is_live(round_id):
            return live.get_accumulated(turn_no)
        return store.get_accumulated(round_id, turn_no)

    @app.get("/api/game_logs")
    def game_logs(round_id: int = Query(0)):
        return []

    @app.get("/api/strategy_config")
    def strategy_config():
        return store.get_strategy_config()

    @app.get("/api/strategy_logs")
    def strategy_logs():
        return []

    index_path = static_dir / "index.html"

    @app.get("/")
    def root():
        return FileResponse(index_path, headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=Path("dump"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--live", action="store_true",
                    help="Poll real game API and expose live round")
    ap.add_argument("--poll-interval", type=float, default=0.25)
    args = ap.parse_args()

    import uvicorn
    static_dir = Path(__file__).parent / "static"
    store = CsvStore(args.dump)
    print(f"Loaded {len(store.rounds)} rounds, {len(store.turns)} turns from {args.dump}")

    live: Optional[LiveStore] = None
    if args.live:
        from config import TOKEN, BASE_URL
        live = LiveStore(BASE_URL, TOKEN, interval=args.poll_interval)
        live.start()
        print(f"Live poller started on {BASE_URL} every {args.poll_interval}s")

    app = build_app(store, live, static_dir)
    print(f"Dashboard at http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
