# Советы игроков → проверяемые гипотезы

Поиск: «одесский покер», «расписной покер», «Joker card game Georgia», «Oh Hell strategy» (2026-11-07).
Прямое чтение страниц (pagat.com, fb.ru, ru.wikipedia.org) было заблокировано сетевой политикой контейнера,
поэтому формулировки взяты из выдачи поисковика (сниппеты). Источники:

- Одесский покер — Википедия: https://ru.wikipedia.org/wiki/Одесский_покер (джокер 7♠ — распространённый вариант; 10 за взятку при точном заказе, 1 за перебор, 5 за пас)
- Расписной покер: правила, особенности и рекомендации — https://fb.ru/article/310301/raspisnoy-poker-pravila-osobennosti-i-rekomendatsii
- Расписной покер (gipsyteam) — https://www.gipsyteam.ru/poker/raspisnoi ; poker.ru — https://poker.ru/raspisnoj-poker-pravila/ ; https://onlinepokerigrat.com/poker/shkola-pokera/pravila-pokera/raspisnoj-poker/ ; https://www.uapoker.info/raspisnoi-poker-pravila-igry-tablitsa ; https://stimul.kiev.ua/articles.htm?a=raspisnoy-poker-dlya-nachinayushchikh-osvoyte-pravila-i-stante-profi ; https://durbetsel.ru/2_odessky_poker.htm
- Joker (Georgia) — https://www.pagat.com/exact/joker.html ; https://groupgames101.com/joker-rules
- Oh Hell — https://www.pagat.com/exact/ohhell.html ; https://en.wikipedia.org/wiki/Oh_Hell ; BGA tips https://en.doc.boardgamearena.com/Tips_ohhell ; https://rarepike.com/how-to-play-oh-hell/ ; https://www.dummies.com/article/home-auto-hobbies/games/card-games/general-card-games/play-card-game-oh-hell-233014/ ; https://entertainment.howstuffworks.com/how-to-play-oh-hell.htm
- Джокер (карточная игра) — Википедия: https://ru.wikipedia.org/wiki/Джокер_(карточная_игра) (хишт; «понял, что не выполнишь — мешай другим»)
- yellowclub.net (Джокер): https://yellowclub.net/page/karty_online/joker ; Amavie Joker: https://joker.amavie.ai/ ; Japanese Joker blog: http://planestrainsmarshrutkas.blogspot.com/2013/04/japanese-joker-georgian-card-game.html
- Oh Hell guides: https://officialgamerules.org/game-rules/oh-hell/ ; https://cardanoir.com/traditional-classic-card-games/oh-hell-card-game-exact-bids-brutal-timing-pure-delight/ ; https://www.mplgames.com/blog/oh-hell-card-game/ ; https://lovecardgames.com/oh-hell
- Исследования ботов: Snelleman (Leiden, Boerenbridge = Oh Hell) https://theses.liacs.nl/pdf/2019-2020-SnellemanT.pdf ; Amend & Zach (Temple) https://cis.temple.edu/~pwang/5603-AI/Project/2021S/Amend_Zach/Amend_Zach_FinalReport.pdf ; Vorsteveld (Maastricht, Rikken) https://project.dke.maastrichtuniversity.nl/games/files/msc/Vorsteveld_thesis.pdf ; AlphaDou (bidding + DMC) https://arxiv.org/abs/2407.10279
- Romanian whist (о невыгоде последнего заказывающего в 1-карточных раздачах) — https://en.wikipedia.org/wiki/Romanian_whist

## Гипотезы (статус и числа — в INSIGHTS.md, раздел «Проверка советов»)

| # | Совет (источник) | Проверяемая гипотеза | Как проверяем |
|---|---|---|---|
| T1 | «Сдающий обычно в проигрыше, особенно в 1-карточных раздачах» (Oh Hell / Romanian whist, Wikipedia) | EV(место сдающего) < EV(остальных мест), разрыв максимален при n_cards=1 | EV по местам в self-play сильного агента, по n_cards; точный эквилибриум для 1 карты |
| T2 | «В ранних раздачах младшие карты неожиданно берут, а тузы бьют козырем; на младшие карты как на проигрышные можно рассчитывать надёжнее, чем на старшие как на выигрышные» (Dummies/HowStuffWorks) | P(взятка | некозырный туз) падает с ростом n_cards медленнее/быстрее…; P(младшая карта не берёт) > P(туз берёт) | калибровочная таблица P(взятка) по классу карты и n_cards |
| T3 | «Считайте козыри — это главный инструмент заказа» (Oh Hell guides) | Число козырей — главный признак в модели заказа (feature importance), ≈ +X взяток за козырь | регрессия взяток на признаки руки, дерево решений |
| T4 | «С джокером ~80% шанс выполнить заказ» (stimul.kiev.ua) | P(заказ выполнен | джокер на руке) ≈ 0.8 и выше, чем без джокера | частоты в self-play |
| T5 | «Заказывайте 2, а не 3, и слушайте заказы остальных» (Oh Hell guide) | Систематически заказывать ниже ожидания (shading −0.5…−1) выгоднее, чем точно по ожиданию | сравнение правил заказа E−δ в duplicate-матчах |
| T6 | «В маленьких раздачах (1–3 карты) пас — самый безопасный заказ со слабой картой» (Oh Hell guide) | Сильный агент пасует чаще, чем heuristic; EV паса со слабой рукой > EV 1 | калибровка + эквилибриум 1 карты |
| T7 | «Джокер — сильнейшая карта, но его сбрасывают младшим, чтобы не взять лишнего»; «перед сбросом говорят «сбрасываю»» | Доля джокеров, сыгранных младшим, у сильного агента > 0; ценность джокера в очках | статистика розыгрыша джокера, EV с джокером vs без |
| T8 | «Заход джокером „по старшим козырям“ вытягивает козыри» (gipsyteam) | Сильный агент при заходе джокером чаще требует козырную масть | частоты заказанной масти при заходе джокером |
| T9 | «Заказывайте в тёмную для удвоения очков» (Yandex Games) | В тёмной есть заказ с положительным EV на каждом месте | EV по заказу в тёмной для каждого места |
| T10 | Асимметрия очков (недобор −10×заказ, перебор +1) ⇒ «лучше недозаказать» (общая логика Oh Hell) | Оптимальный заказ < ожидаемого числа взяток; доля переборов > доли недоборов у сильного агента | распределение (взятки − заказ) |
| T11 | «Следите за заказами других — и в заказе, и в розыгрыше» | Агент, учитывающий заказы соперников (PIMC с пересэмплированием по заказам), сильнее того же без этого | ablation sigma=∞ |
| T12 | «Ход с туза, когда нужны взятки; сбрасывать самую младшую» (fb.ru) | Это правила heuristic-агента; сильный агент от них отклоняется в X% случаев | доля совпадений ходов heuristic и лучшего агента |
| T13 | «Понял, что не выполнишь заказ — мешай другим: навязывай лишнюю взятку / забирай нужную» (ru.wikipedia, Джокер); «подкормить взяткой того, кому она не нужна — половина мастерства» (Amavie) | После «сгоревшего» заказа сильный агент снижает очки соперников сильнее, чем heuristic | средние очки соперников в раздачах, где у агента недобор неизбежен (предварительно) |
| T14 | «Первую взятку бери тузом, запоминай вышедшие карты» (yellowclub) | Сильный агент, нуждаясь во взятках, чаще заходит с некозырного туза в 1-й взятке | частота ходов |
| T15 | «Если заказы соперников в сумме больше числа карт — заказывай осторожно; если меньше — агрессивно» (rarepike, mplgames) | При фиксированной руке заказ сильного агента убывает с суммой предыдущих заказов | OLS заказа на признаки руки + сумму предыдущих заказов |
| T16 | «Пас — оружие, но не лёгкий: соперники вгружают взятки» (cardanoir) | P(пас выполнен) < 1 и падает с n_cards; EV паса | калибровка |
| T17 | «Самое трудное — перестать брать взятки после выполнения заказа: сбрасывайте старшие» (cardanoir) | Перебор — главный источник промахов сильного агента (доля переборов среди промахов) | распределение (взятки − заказ) |
| T18 | Исследования ботов Oh Hell / Boerenbridge: Monte-Carlo-агенты сильнее rule-based, но rule-based заказ может быть лучшим (Rikken) | В нашей игре: PIMC vs heuristic vs DMC; вклад заказа vs розыгрыша | лидерборд, гибриды «заказ A + розыгрыш B» |
