# Одесский покер: движок, агенты, обучение и инсайды

Окружение для взяточной игры «Одесский покер» (расписной / студенческий покер, родственник Oh Hell и
грузинского Joker), набор агентов (random, heuristic, PIMC, точный эквилибриум для 1 карты, Deep Monte Carlo),
duplicate-турниры с Elo и анализ стратегии.

**Главное — в [INSIGHTS.md](INSIGHTS.md)** (как играть), **[leaderboard.md](leaderboard.md)** (сила агентов),
[DECISIONS.md](DECISIONS.md) (все спорные решения), [PROGRESS.md](PROGRESS.md), [research/tips.md](research/tips.md).

## Быстрый старт

```bash
pip install numpy pyyaml pytest torch scikit-learn
python -m pytest -q tests          # тесты движка/env/eval
python scripts/bench.py            # скорость
```

### Сыграть самому

```bash
python -m play.cli --opponents store:best heuristic heuristic          # полная партия (28 раздач)
python -m play.cli --deal 5 --opponents pimc:samples=24                # одна раздача по 5 карт
```
Вы вводите номер варианта. Джокер: `JOKER-high/low` — старший/младший, `JOKER-lead-high(♠)` — заход «по старшим пикам»,
`JOKER-lead-low(♦)` — заход джокером как младшей бубной.

Спеки агентов (везде одинаковые): `random`, `heuristic`, `heuristic:bid_noise=0.7`, `pimc:samples=24`,
`learned:path=runs/dmc/dmc_v1/qnet.pt`, `store:<name>[/<version>]` (см. `agents_store/`).

### Турнир / лидерборд

```bash
python -m eval.leaderboard --agents heuristic pimc:samples=16 store:best --games 100 --pair-games 50
```
Duplicate-формат: одни и те же расклады, кандидат по очереди на каждом месте против 3×field;
Δ = очки за раздачу относительно field на тех же раскладах, 95% bootstrap CI. Pairwise ABAB/BABA → Bradley–Terry Elo.

### Дообучить

```bash
python -m train.dmc --name dmc_v3 --minutes 120 --actors 3 --init agents_store/best/1/qnet.pt
python -c "from registry.store import save_agent; save_agent('dmc_v3','1','learned',{},{'qnet.pt':'runs/dmc/dmc_v3/qnet.pt'})"
```
Логи: `runs/dmc/<name>/log.csv`, чекпоинты `ckpt_*.pt`.

### Анализ

```bash
python -m analysis.one_card --iters 40                       # точный эквилибриум заказа для 1 карты
python -m analysis.collect --specs store:best --deals 40000 --out runs/data/best.npz
python -m analysis.calibration runs/data/best.npz            # таблицы калибровки
python -m analysis.experiments all                           # эксперименты для INSIGHTS.md
```

## Структура

```
engine/    rules.py (rules.yaml -> Rules), core.py (state/legal/step/score), obs.py (информация игрока), runner.py
env/       aec.py (PettingZoo-AEC-подобный API, маски), features.py (вектор наблюдения)
agents/    random, heuristic ("типичный человек"), solver (max^n точный розыгрыш), pimc, small_exact (1 карта),
           learned (DMC Q-сеть), hybrid (заказ одним агентом, розыгрыш другим)
registry/  store.py — спеки, agents_store/<name>/<version>/{config.json, meta.json, weights}
eval/      harness.py (duplicate, pairwise, BT-Elo, bootstrap), leaderboard.py
train/     dmc.py (Deep Monte Carlo self-play, лига)
analysis/  one_card.py, collect.py, calibration.py, experiments.py, distill.py
play/      cli.py
tests/     property-тесты движка, env, eval
```

## Варианты правил (rules.yaml)

| флаг | по умолчанию | варианты | смысл |
|---|---|---|---|
| `n_players` | 4 | 3–6 | max карт = 36 // n |
| `deck.jokers` | `["7S"]` | `[]`, `["6S"]`, два джокера | какие карты — джокеры |
| `schedule.ascending/descending` | true | false | 1..max−1 и max−1..1 |
| `schedule.max_repeats` | null (= n) | число | раздач по max подряд |
| `schedule.special` | n бескозырок + n тёмных | список `{type, count, n_cards}` | спецраунды |
| `trump.joker_turned` | notrump | — | открыт джокер → бескозырка |
| `trump.no_cards_left` | notrump | `dealer_last` | карт не осталось → бескозырка / масть последней карты сдающего |
| `bidding.dealer_hook` | true | false | сдающий не может сделать сумму заказов = n_cards |
| `bidding.dark_sees_trump` | true | false | в тёмной козырь виден до заказа |
| `play.must_follow` | true | false | обязательно в масть |
| `play.must_trump` | true | false | нет масти → обязательно козырь |
| `play.joker_any_time` | true | false | джокер можно играть, даже имея масть |
| `play.joker_lead_low` | true | false | заход джокером как младшей картой названной масти |
| `play.joker_lead_demand_highest` | true | false | заход «старший X» требует старшую карту X |
| `scoring.exact_per_trick` | 10 | | ровно: 10×заказ |
| `scoring.pass_exact` | 5 | | пас выполнен |
| `scoring.over_per_trick` | 1 | | перебор: за каждую лишнюю |
| `scoring.under_per_trick` / `under_mode` | −10 / `bid` | `missing` | недобор: −10×заказ или −10×недостача |
| `scoring.dark_multiplier` | 2 | | тёмная ×2 |

Не реализовано: «после двух пасов подряд обязан заказывать», премии за пульку, штрафы за ренонс (движок не допускает нелегальных ходов).
