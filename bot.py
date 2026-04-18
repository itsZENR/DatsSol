"""
DatsSol bot v1.0
Main loop: get state → decide → submit commands.
"""

import time
import logging
import sys
from strategies import BotState, select_and_merge
from client import get_arena, post_command, get_logs
from config import POLL_INTERVAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bot")


def run():
    bot = BotState()
    last_turn = -1

    log.info("Bot started")

    while True:
        try:
            state = get_arena()

            if state.turn_no == last_turn:
                time.sleep(POLL_INTERVAL)
                continue

            last_turn = state.turn_no
            log.info(
                "Turn %d | plantations=%d | connected=%d | enemies=%d | beavers=%d | hp_points=%d",
                state.turn_no,
                len(state.plantations),
                len(state.connected_plantations),
                len(state.enemies),
                len(state.beavers),
                state.upgrades.points,
            )

            if not state.main_plantation:
                log.warning("No HQ found, waiting for respawn")
                time.sleep(1.0)
                continue

            decision = select_and_merge(state, bot)
            payload = decision.to_payload()

            if payload:
                resp = post_command(payload)
                if resp.get("errors"):
                    for err in resp["errors"]:
                        if "already submitted" not in err:
                            log.warning("Command error: %s", err)

            # Sleep until just before next turn
            sleep_time = max(0.05, state.next_turn_in - 0.1)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Stopped by user")
            break
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
            time.sleep(1.0)


if __name__ == "__main__":
    run()
