# PROGRESS

## Статус фаз
- [x] Фаза 1: движок (engine/), rules.yaml, тесты (tests/test_engine.py, 17 зелёных), бенчмарк, random + heuristic агенты.
- [x] Фаза 2: env/aec.py (AEC-подобный API + features.py, obs_dim=341 для 4 игроков), registry/store.py (спеки агентов + agents_store), eval/harness.py (duplicate vs-field, pairwise ABAB/BABA, Bradley–Terry Elo, bootstrap CI), eval/leaderboard.py
- [x] Фаза 3: точный эквилибриум 1-карточных раздач (analysis/one_card.py, полный перебор 45M раскладов, fictitious play 40 итераций, exploitability ≤0.02 очк/раздачу); точный max^n солвер розыгрыша (agents/solver.py) для ≤4 карт
- [x] Фаза 4: PIMC (agents/pimc.py): +3.9 [3.2, 4.5] очк/раздачу vs heuristic-поле (40 партий, samples=16), ~3.3 с/партию
- [~] Фаза 5: DMC (train/dmc.py) dmc_v1 идёт 110 мин, ~14k сэмплов/с; чекпоинты: 8.4M −0.41, 16.5M +1.47 vs heuristic
- [ ] Фаза 6: инсайды, README

## Числа
- Скорость (scripts/bench.py, 1 ядро): сырой движок 10.2k раздач(9 карт)/с, 409k действий/с; полная партия (28 раздач) random-агентами 300 партий/с, heuristic 160 партий/с.
- Первый лидерборд (100 партий): random −39.7 очк/раздачу vs heuristic; heuristic с шумом заказа σ=0.7 −3.9 [−4.2,−3.5].
- 1-карточный эквилибриум: EV по местам [4.26, 4.99, 5.21, −1.81]; против heuristic-поля даёт +0.76 [0.68, 0.84] очк/раздачу (3000 раздач).
- heuristic2 vs heuristic-поле: +1.96 [1.68, 2.24] (100 партий); вклад розыгрыша +1.70, заказа +0.57.
- Против поля heuristic2 (сильные соперники): DMC v1@34M +1.42 [1.08, 1.77] (60 партий); PIMC(16, роллауты h2) +4.22 [3.54, 4.90]; PIMC(16, роллауты h1) +2.01 [1.47, 2.54] (30 партий).
