# Leaderboard

Field = `heuristic`; duplicate format (same deals, candidate rotated through all 4 seats vs 3×field, compared to all-field baseline on identical deals). Δ = points per deal vs field, 95% bootstrap CI over 100 full games (28 deals each). Elo = Bradley–Terry over pairwise duplicate games (ABAB/BABA seatings, 100 game-pairs per pair), anchored heuristic=1500.

| agent | Δ vs field (pts/deal) | 95% CI | Elo |
|---|---|---|---|
| `heuristic` | +0.00 | [+0.00, +0.00] | 1500 |
| `heuristic:bid_noise=0.7` | -3.86 | [-4.21, -3.49] | 1094 |
| `random` | -39.72 | [-40.43, -39.02] | 278 |

## Pairwise (A − B, pts/deal; win-rate of A by game total)

| A | B | A−B | 95% CI | A win-rate |
|---|---|---|---|---|
| `random` | `heuristic` | -38.57 | [-39.28, -37.85] | 0.00 |
| `random` | `heuristic:bid_noise=0.7` | -35.57 | [-36.33, -34.85] | 0.00 |
| `heuristic` | `heuristic:bid_noise=0.7` | +3.08 | [+2.64, +3.55] | 0.92 |

_generated in 4s_
