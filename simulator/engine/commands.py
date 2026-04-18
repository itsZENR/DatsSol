from dataclasses import dataclass, field
from typing import Optional
from simulator.engine.types import Pos, UpgradeName


@dataclass
class Command:
    author: Pos
    relay: Pos
    target: Pos

    def to_path(self) -> list[list[int]]:
        return [list(self.author), list(self.relay), list(self.target)]

    @classmethod
    def from_path(cls, path: list[list[int]]) -> "Command":
        a, r, t = path
        return cls(author=(a[0], a[1]), relay=(r[0], r[1]), target=(t[0], t[1]))


@dataclass
class CommandSet:
    commands: list[Command] = field(default_factory=list)
    upgrade: Optional[UpgradeName] = None
    relocate: list[Pos] = field(default_factory=list)

    def to_payload(self) -> dict:
        payload: dict = {}
        if self.commands:
            payload["command"] = [{"path": c.to_path()} for c in self.commands]
        if self.upgrade is not None:
            payload["plantationUpgrade"] = self.upgrade.value
        if self.relocate:
            payload["relocateMain"] = [list(p) for p in self.relocate]
        if not payload:
            payload["plantationUpgrade"] = ""
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> "CommandSet":
        cmds = [Command.from_path(c["path"]) for c in payload.get("command", [])]
        upg_raw = payload.get("plantationUpgrade", "")
        upg = UpgradeName(upg_raw) if upg_raw else None
        relo_raw = payload.get("relocateMain", [])
        relo = [(p[0], p[1]) for p in relo_raw]
        return cls(commands=cmds, upgrade=upg, relocate=relo)
