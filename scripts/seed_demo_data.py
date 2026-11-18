#!/usr/bin/env python3
"""
seed_demo_data.py — fill data/ with sample stats so the dashboard has
something to show before you have a week of real history.

Writes data/stats.json and data/roast_history.json in the same format the
app produces. The events are made up. Existing files are left alone unless
--force is given.

Usage (from the repo root):
    python scripts/seed_demo_data.py [--days 7] [--seed 42] [--force]
"""

import argparse
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = ROOT / "data" / "stats.json"
HISTORY_PATH = ROOT / "data" / "roast_history.json"

DEMO_EVENTS = [
    "Algorithms lab report",
    "Thesis draft",
    "Embedded systems lecture",
    "Physics problem set",
    "Team stand-up",
    "Read chapter 4",
]

# Distractions cluster in the afternoon slump and late evening.
HOURS = [9, 10, 11, 13, 14, 14, 15, 15, 16, 17, 20, 21, 21, 22, 22, 23]


class _Defaults(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def build(days: int, rng: random.Random, templates: list) -> tuple:
    now = datetime.now()
    daily, kill_log, worst_hours, history = {}, [], {}, []
    longest_streak = 0.0
    previous = None

    for offset in range(days - 1, -1, -1):
        day = date.today() - timedelta(days=offset)
        kills = []
        for _ in range(rng.randint(3, 10)):
            ts = datetime.combine(day, datetime.min.time()) + timedelta(
                hours=rng.choice(HOURS), minutes=rng.randint(0, 59), seconds=rng.randint(0, 59)
            )
            if ts < now:
                kills.append((ts, round(rng.uniform(2.0, 14.0), 1)))
        kills.sort()

        for ts, duration in kills:
            kill_log.append({"date": day.isoformat(), "time": ts.isoformat(), "duration": duration})
            hour = str(ts.hour)
            worst_hours[hour] = worst_hours.get(hour, 0) + 1
            if previous is not None:
                longest_streak = max(longest_streak, (ts - previous).total_seconds())
            previous = ts

            event, next_event = rng.sample(DEMO_EVENTS, 2)
            kill_count = sum(1 for k in kill_log if k["date"] == day.isoformat())
            text = rng.choice(templates).format_map(_Defaults(
                event=event, next_event=next_event,
                minutes_until=rng.randint(5, 50), kill_count=kill_count,
            ))
            history.append({
                "timestamp": (ts + timedelta(seconds=3)).isoformat(),
                "roast": text,
                "current_event": event,
                "next_event": next_event,
                "kill_count": kill_count,
            })

        if kills:
            daily[day.isoformat()] = {
                "kill_count": len(kills),
                "total_duration": round(sum(d for _, d in kills), 1),
                "kills": [{"time": ts.isoformat(), "duration": d} for ts, d in kills],
            }

    stats = {
        "daily": daily,
        "total_kills": len(kill_log),
        "longest_streak_seconds": round(longest_streak, 1),
        "worst_hours": worst_hours,
        "kill_log": kill_log,
    }
    return stats, history


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--days", type=int, default=7, help="days of history to generate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--force", action="store_true", help="overwrite existing data files")
    args = parser.parse_args()

    if not args.force and (STATS_PATH.exists() or HISTORY_PATH.exists()):
        raise SystemExit("data/ already has stats. Use --force to replace them.")

    with open(ROOT / "config" / "roasts.yaml", encoding="utf-8") as f:
        templates = yaml.safe_load(f)["roasts"]

    stats, history = build(args.days, random.Random(args.seed), templates)

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Wrote {stats['total_kills']} sample detections over {args.days} days to data/")


if __name__ == "__main__":
    main()
