# PROGRESS

## Статус фаз
- [x] Фаза 1: движок (engine/), rules.yaml, тесты (tests/test_engine.py, 17 зелёных), бенчмарк, random + heuristic агенты.
- [x] Фаза 2: env/aec.py (AEC-подобный API + features.py, obs_dim=341 для 4 игроков), registry/store.py (спеки агентов + agents_store), eval/harness.py (duplicate vs-field, pairwise ABAB/BABA, Bradley–Terry Elo, bootstrap CI), eval/leaderboard.py
- [ ] Фаза 3: точные решения малых раздач
- [ ] Фаза 4: PIMC
- [ ] Фаза 5: обучение
- [ ] Фаза 6: инсайды, README

## Числа
- Скорость (scripts/bench.py, 1 ядро): сырой движок 10.2k раздач(9 карт)/с, 409k действий/с; полная партия (28 раздач) random-агентами 300 партий/с, heuristic 160 партий/с.
- Первый лидерборд (100 партий): random −39.7 очк/раздачу vs heuristic; heuristic с шумом заказа σ=0.7 −3.9 [−4.2,−3.5].
