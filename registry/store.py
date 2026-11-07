"""Agent registry.

Spec strings (used everywhere: eval, play, analysis):
    random | heuristic | heuristic:bid_noise=0.7 | pimc:samples=24 | store:<name>[/<version>]
Store layout: agents_store/<name>/<version>/{config.json, meta.json, weights files}
    config.json = {"kind": "<agent kind>", "kwargs": {...}}   (weights paths relative to the version dir)
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "agents_store")


def _parse_kwargs(s: str) -> dict:
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def _kinds():
    from agents.heuristic import HeuristicAgent
    from agents.random_agent import RandomAgent
    from agents.heuristic2 import SmartHeuristicAgent
    kinds = {"random": RandomAgent, "heuristic": HeuristicAgent, "heuristic2": SmartHeuristicAgent}
    try:
        from agents.pimc import PIMCAgent
        kinds["pimc"] = PIMCAgent
    except ImportError:
        pass
    try:
        from agents.learned import LearnedAgent
        kinds["learned"] = LearnedAgent
    except ImportError:
        pass
    from agents.wrappers import WrapAgent
    kinds["wrap"] = WrapAgent
    from agents.hybrid import HybridAgent
    kinds["hybrid"] = HybridAgent
    try:
        from agents.small_exact import SmallExactAgent
        kinds["exact"] = SmallExactAgent
    except ImportError:
        pass
    return kinds


def latest_version(name: str) -> str:
    d = os.path.join(STORE, name)
    vs = sorted(os.listdir(d), key=lambda v: (len(v), v))
    return vs[-1]


def make_agent(spec: str, seed: int = 0, rules=None):
    kind, _, rest = spec.partition(":")
    if kind == "store":
        name, _, ver = rest.partition("/")
        ver = ver or latest_version(name)
        vdir = os.path.join(STORE, name, ver)
        with open(os.path.join(vdir, "config.json")) as f:
            cfg = json.load(f)
        kw = dict(cfg.get("kwargs", {}))
        kw["model_dir"] = vdir
        ag = _kinds()[cfg["kind"]](seed=seed, **kw)
        ag.name = f"{name}/{ver}"
        return ag
    kw = _parse_kwargs(rest)
    ag = _kinds()[kind](seed=seed, **kw)
    ag.name = spec
    return ag


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def save_agent(name: str, version: str, kind: str, kwargs: dict, files: dict | None = None,
               meta: dict | None = None, rules=None) -> str:
    """files: {dest_name: src_path} copied into the version dir."""
    vdir = os.path.join(STORE, name, version)
    os.makedirs(vdir, exist_ok=True)
    for dst, src in (files or {}).items():
        shutil.copy(src, os.path.join(vdir, dst))
    with open(os.path.join(vdir, "config.json"), "w") as f:
        json.dump({"kind": kind, "kwargs": kwargs}, f, indent=2)
    m = dict(git_sha=git_sha(), date=datetime.datetime.now().isoformat(timespec="seconds"),
             rules_hash=rules.hash if rules else None)
    m.update(meta or {})
    with open(os.path.join(vdir, "meta.json"), "w") as f:
        json.dump(m, f, indent=2)
    return vdir


def update_meta(name: str, version: str, **kv):
    p = os.path.join(STORE, name, version, "meta.json")
    with open(p) as f:
        m = json.load(f)
    m.update(kv)
    with open(p, "w") as f:
        json.dump(m, f, indent=2)


def list_agents() -> list:
    out = []
    if not os.path.isdir(STORE):
        return out
    for name in sorted(os.listdir(STORE)):
        for ver in sorted(os.listdir(os.path.join(STORE, name))):
            out.append(f"store:{name}/{ver}")
    return out
