# Leaderboard

Field = `heuristic2`; duplicate format (same deals, candidate rotated through all 4 seats vs 3×field, compared to all-field baseline on identical deals). Δ = points per deal vs field, 95% bootstrap CI over 60 full games (28 deals each). Elo = Bradley–Terry over pairwise duplicate games (ABAB/BABA seatings, 30 game-pairs per pair), anchored heuristic2=1500.

| agent | Δ vs field (pts/deal) | 95% CI | Elo |
|---|---|---|---|
| `store:pimc_h2c` | +4.41 | [+3.91, +4.88] | 1957 |
| `store:dmc_v3` | +3.79 | [+3.52, +4.06] | 2094 |
| `store:dmc_v2` | +3.57 | [+3.33, +3.82] | 2003 |
| `store:dmc_v1` | +2.02 | [+1.69, +2.34] | 1793 |
| `store:heuristic2c` | +1.03 | [+0.85, +1.21] | 1616 |
| `heuristic2` | +0.00 | [+0.00, +0.00] | 1500 |
| `heuristic` | -1.50 | [-1.82, -1.17] | 1254 |

## Pairwise (A − B, pts/deal; win-rate of A by game total)

| A | B | A−B | 95% CI | A win-rate |
|---|---|---|---|---|
| `heuristic` | `heuristic2` | -2.14 | [-2.81, -1.49] | 0.13 |
| `heuristic` | `store:heuristic2c` | -2.82 | [-3.62, -2.09] | 0.10 |
| `heuristic` | `store:dmc_v1` | -3.47 | [-4.12, -2.86] | 0.00 |
| `heuristic` | `store:dmc_v2` | -4.58 | [-5.25, -3.88] | 0.03 |
| `heuristic` | `store:dmc_v3` | -4.77 | [-5.39, -4.12] | 0.00 |
| `heuristic` | `store:pimc_h2c` | -4.28 | [-5.03, -3.52] | 0.03 |
| `heuristic2` | `store:heuristic2c` | -0.75 | [-1.16, -0.33] | 0.30 |
| `heuristic2` | `store:dmc_v1` | -2.03 | [-2.67, -1.36] | 0.17 |
| `heuristic2` | `store:dmc_v2` | -3.12 | [-3.66, -2.64] | 0.00 |
| `heuristic2` | `store:dmc_v3` | -3.34 | [-3.79, -2.90] | 0.00 |
| `heuristic2` | `store:pimc_h2c` | -3.42 | [-4.23, -2.67] | 0.07 |
| `store:heuristic2c` | `store:dmc_v1` | -1.09 | [-1.71, -0.41] | 0.20 |
| `store:heuristic2c` | `store:dmc_v2` | -2.44 | [-3.03, -1.91] | 0.07 |
| `store:heuristic2c` | `store:dmc_v3` | -2.65 | [-3.07, -2.22] | 0.03 |
| `store:heuristic2c` | `store:pimc_h2c` | -1.79 | [-2.59, -1.04] | 0.17 |
| `store:dmc_v1` | `store:dmc_v2` | -1.95 | [-2.51, -1.43] | 0.10 |
| `store:dmc_v1` | `store:dmc_v3` | -2.16 | [-2.90, -1.44] | 0.17 |
| `store:dmc_v1` | `store:pimc_h2c` | -1.67 | [-2.70, -0.69] | 0.30 |
| `store:dmc_v2` | `store:dmc_v3` | -0.86 | [-1.38, -0.27] | 0.23 |
| `store:dmc_v2` | `store:pimc_h2c` | +0.44 | [-0.30, +1.14] | 0.57 |
| `store:dmc_v3` | `store:pimc_h2c` | +0.66 | [-0.15, +1.51] | 0.57 |

_generated in 128s_

## Как читать

- **Сильнейший агент по Elo — `store:dmc_v3` (= `store:best`)**: DMC self-play, раздельные сети заказа и розыгрыша,
  ~237M сэмплов (v1 → v2 → v3), ~5 ч CPU. Быстрый (~0.1 мс на решение) — им удобно играть самому.
- `pimc_h2c` лучше всех против поля heuristic2, но это частично «знание соперника»: его роллауты — та же эвристика, что и
  соперники. Очно `dmc_v3 − pimc_h2c` = +0.66 [−0.15, +1.51] очк/раздачу — неразличимы.
- Каждая следующая версия DMC значимо сильнее предыдущей: v2−v1 = +1.95 [1.43, 2.51], v3−v2 = +0.86 [0.27, 1.38].
- `heuristic` (модель «типичного человека») проигрывает сильным агентам 3.5–4.8 очка за раздачу (разница ≈ 100–130 очков за
  партию из 28 раздач, не итоговый счёт); `heuristic2` («сильный клубный игрок») — 2.0–3.4.
- Прочие замеры (PIMC с NN-роллаутами, гибриды, абляции) — в PROGRESS.md.
