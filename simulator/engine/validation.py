from simulator.engine.state import EngineState
from simulator.engine.commands import Command
from simulator.engine.spatial import chebyshev


def is_valid_command(state: EngineState, cmd: Command) -> bool:
    """Geometric validity: author/relay are owned+connected,
    relay within SR of author, target within AR of relay."""
    pos_to_id = {
        tuple(p.position): pid
        for pid, p in state.plantations.items()
    }
    author_id = pos_to_id.get(cmd.author)
    relay_id = pos_to_id.get(cmd.relay)
    if author_id is None or relay_id is None:
        return False
    if author_id not in state.connected_ids:
        return False
    if relay_id not in state.connected_ids:
        return False
    if chebyshev(cmd.author, cmd.relay) > state.effective_sr():
        return False
    if chebyshev(cmd.relay, cmd.target) > state.config.ar:
        return False
    return True
