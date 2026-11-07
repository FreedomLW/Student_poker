# Одесский покер: движок, агенты, обучение и инсайды

Окружение для взяточной игры «Одесский покер» (расписной / студенческий покер, родственник Oh Hell и
грузинского Joker), набор агентов (random, heuristic, PIMC, точный эквилибриум для 1 карты, Deep Monte Carlo),
duplicate-турниры с Elo и анализ стратегии.

**Главное — в [INSIGHTS.md](INSIGHTS.md)** (как играть), **[leaderboard.md](leaderboard.md)** (сила агентов),
[DECISIONS.md](DECISIONS.md) (все спорные решения), [PROGRESS.md](PROGRESS.md), [research/tips.md](research/tips.md).

## Результаты в двух словах

| агент | что это | Elo | Δ vs heuristic2, очк/раздачу |
|---|---|---|---|
| `store:best` = `store:dmc_v3` | DMC self-play, раздельные сети заказа и розыгрыша | **2094** | +3.79 [3.52, 4.06] |
| `store:dmc_v2` | то же, предыдущая версия | 2003 | +3.57 |
| `store:pimc_h2c` | PIMC, 16 миров, роллауты heuristic2c, точный эндшпиль | 1957 | +4.41 (знает модель соперника) |
| `store:dmc_v1` | DMC, одна сеть | 1793 | +2.02 |
| `store:heuristic2c` | «сильный клубный игрок» + калиброванный заказ | 1616 | +1.03 |
| `heuristic2` | «сильный клубный игрок» | 1500 | 0 |
| `heuristic` | «типичный человек» | 1254 | −1.50 |

Ключевые выводы (подробно, с CI — в [INSIGHTS.md](INSIGHTS.md)): заказывайте на ~0.4 взятки меньше ожидания;
козырь ≈ +0.65 взятки, некозырный туз в коротких раздачах почти ничего не стоит; джокер ≈ +6…+20 очков и +25…+40 п.п.
к шансу выполнить заказ; сдающий теряет 6.5 очка в 1-карточной раздаче из-за правила сдающего; в тёмной — заказ 2.

## Быстрый старт

```bash
pip install numpy pyyaml pytest torch scikit-learn
python -m pytest -q tests          # тесты движка/env/eval
python scripts/bench.py            # скорость
```

### Сыграть самому

```bash
python -m play.cli --opponents store:best                              # полная партия (28 раздач) против трёх лучших агентов
python -m play.cli --opponents store:best heuristic heuristic          # смешанный стол
python -m play.cli --deal 5 --opponents pimc:samples=24                # одна раздача по 5 карт
```
Вы вводите номер варианта. Джокер: `JOKER-high/low` — старший/младший, `JOKER-lead-high(♠)` — заход «по старшим пикам»,
`JOKER-lead-low(♦)` — заход джокером как младшей бубной.

Спеки агентов (везде одинаковые): `random`, `heuristic`, `heuristic:bid_noise=0.7`, `pimc:samples=24`,
`learned:path=runs/dmc/dmc_v1/qnet.pt`, `store:<name>[/<version>]` (см. `agents_store/`).

### Турнир / лидерборд

```bash
python -m eval.leaderboard --agents heuristic heuristic2 store:dmc_v3 store:pimc_h2c --field heuristic2 --games 60 --pair-games 30
```
Duplicate-формат: одни и те же расклады, кандидат по очереди на каждом месте против 3×field;
Δ = очки за раздачу относительно field на тех же раскладах, 95% bootstrap CI. Pairwise ABAB/BABA → Bradley–Terry Elo.

### Дообучить

```bash
# продолжить обучение лучшей сети (раздельные сети заказа и розыгрыша)
python -m train.dmc --name dmc_v4 --minutes 120 --actors 3 --separate --lr 5e-5 --p-heur 0.3 --p-heur2 0.8 \
    --init-bid agents_store/dmc_v3/1/bidnet.pt --init-play agents_store/dmc_v3/1/playnet.pt
python scripts/eval_ckpts.py runs/dmc/dmc_v4 60 4 heuristic2   # все чекпоинты vs поле heuristic2
python -c "from registry.store import save_agent; save_agent('dmc_v4','1','learned',{},{'bidnet.pt':'runs/dmc/dmc_v4/bidnet.pt','playnet.pt':'runs/dmc/dmc_v4/playnet.pt'})"
```
Логи: `runs/dmc/<name>/log.csv`, чекпоинты `ckpt_*.pt`.

### Анализ

```bash
python -m analysis.one_card --iters 40                       # точный эквилибриум заказа для 1 карты
python -m analysis.collect --specs store:best --deals 40000 --out runs/data/best.npz
python -m analysis.calibration runs/data/best.npz            # таблицы калибровки
python -m analysis.tables runs/data/best.npz                 # EV по местам, джокер, поведение заказа, тёмная
PROCS=4 python -m analysis.experiments all --best store:dmc_v3  # duplicate-эксперименты: сдвиг заказа, джокер, тёмная, туз
python -m analysis.actions --spec store:dmc_v3                # пошаговая статистика ходов
python -m analysis.distill runs/data/best.npz                 # дерево решений для заказа
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

**По умолчанию — правила компании** (с 2026-11-07): козырь всегда бубны; джокер 7♠ можно сыграть в любой свой ход;
заходя с него — только «старшую масти X»; в ответ — как джокер (бьёт всё) или как обычная 7♠.
Прежние «классические» правила (открытый козырь, бескозырки, джокер-младший = сброс): `rules_variants/classic.yaml`
(`load_rules("rules_variants/classic.yaml")`). Результаты в INSIGHTS.md §§0–8 получены на классических правилах,
раздел «Правила компании» — на новых.

| флаг | по умолчанию | варианты | смысл |
|---|---|---|---|
| `n_players` | 4 | 3–6 | max карт = 36 // n |
| `deck.jokers` | `["7S"]` | `[]`, `["6S"]`, два джокера | какие карты — джокеры |
| `schedule.ascending/descending` | true | false | 1..max−1 и max−1..1 |
| `schedule.max_repeats` | null (= n) | число | раздач по max подряд |
| `schedule.special` | n тёмных | список `{type, count, n_cards}` (`notrump`, `dark`) | спецраунды |
| `trump.mode` | **fixed** | `turned` | козырь всегда `fixed_suit` / следующая открытая карта |
| `trump.fixed_suit` | **D** (бубны) | S, C, H | |
| `trump.joker_turned` | notrump | — | (для turned) открыт джокер → бескозырка |
| `trump.no_cards_left` | notrump | `dealer_last` | карт не осталось → бескозырка / масть последней карты сдающего |
| `bidding.dealer_hook` | true | false | сдающий не может сделать сумму заказов = n_cards |
| `bidding.dark_sees_trump` | true | false | в тёмной козырь виден до заказа |
| `play.must_follow` | true | false | обязательно в масть |
| `play.must_trump` | true | false | нет масти → обязательно козырь |
| `play.joker_any_time` | true | false | джокер можно играть, даже имея масть |
| `play.joker_lead_low` | **false** | true | заход джокером как младшей картой названной масти |
| `play.joker_follow_low` | **as_card** | discard | в ответ «не джокером»: как обычная 7♠ (может взять) / гарантированный сброс |
| `play.joker_lead_demand_highest` | true | false | заход «старший X» требует старшую карту X |
| `scoring.exact_per_trick` | 10 | | ровно: 10×заказ |
| `scoring.pass_exact` | 5 | | пас выполнен |
| `scoring.over_per_trick` | 1 | | перебор: за каждую лишнюю |
| `scoring.under_per_trick` / `under_mode` | −10 / `bid` | `missing` | недобор: −10×заказ или −10×недостача |
| `scoring.dark_multiplier` | 2 | | тёмная ×2 |

Не реализовано: «после двух пасов подряд обязан заказывать», премии за пульку, штрафы за ренонс (движок не допускает нелегальных ходов).
