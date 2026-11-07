
### depth 3: exact agreement 0.555, within ±1 0.939 (test n=13716)

```
|--- hcp <= 6.50
|   |--- joker <= 0.50
|   |   |--- n_trump <= 0.50
|   |   |   |--- class: 0
|   |   |--- n_trump >  0.50
|   |   |   |--- class: 1
|   |--- joker >  0.50
|   |   |--- hcp <= 3.50
|   |   |   |--- class: 1
|   |   |--- hcp >  3.50
|   |   |   |--- class: 2
|--- hcp >  6.50
|   |--- hcp <= 12.50
|   |   |--- joker <= 0.50
|   |   |   |--- class: 2
|   |   |--- joker >  0.50
|   |   |   |--- class: 3
|   |--- hcp >  12.50
|   |   |--- joker <= 0.50
|   |   |   |--- class: 3
|   |   |--- joker >  0.50
|   |   |   |--- class: 5
```

### depth 4: exact agreement 0.629, within ±1 0.977 (test n=13716)

```
|--- hcp <= 6.50
|   |--- joker <= 0.50
|   |   |--- n_trump <= 0.50
|   |   |   |--- is_trump_deal <= 0.50
|   |   |   |   |--- class: 0
|   |   |   |--- is_trump_deal >  0.50
|   |   |   |   |--- class: 0
|   |   |--- n_trump >  0.50
|   |   |   |--- n_trump <= 1.50
|   |   |   |   |--- class: 0
|   |   |   |--- n_trump >  1.50
|   |   |   |   |--- class: 1
|   |--- joker >  0.50
|   |   |--- hcp <= 3.50
|   |   |   |--- n_trump <= 1.50
|   |   |   |   |--- class: 1
|   |   |   |--- n_trump >  1.50
|   |   |   |   |--- class: 2
|   |   |--- hcp >  3.50
|   |   |   |--- voids <= 1.50
|   |   |   |   |--- class: 2
|   |   |   |--- voids >  1.50
|   |   |   |   |--- class: 1
|--- hcp >  6.50
|   |--- hcp <= 12.50
|   |   |--- joker <= 0.50
|   |   |   |--- hcp <= 9.50
|   |   |   |   |--- class: 1
|   |   |   |--- hcp >  9.50
|   |   |   |   |--- class: 2
|   |   |--- joker >  0.50
|   |   |   |--- n_cards <= 8.50
|   |   |   |   |--- class: 2
|   |   |   |--- n_cards >  8.50
|   |   |   |   |--- class: 3
|   |--- hcp >  12.50
|   |   |--- joker <= 0.50
|   |   |   |--- is_trump_deal <= 0.50
|   |   |   |   |--- class: 3
|   |   |   |--- is_trump_deal >  0.50
|   |   |   |   |--- class: 2
|   |   |--- joker >  0.50
|   |   |   |--- n_cards <= 8.50
|   |   |   |   |--- class: 3
|   |   |   |--- n_cards >  8.50
|   |   |   |   |--- class: 5
```

### depth 6: exact agreement 0.738, within ±1 0.989 (test n=13716)


### depth 8: exact agreement 0.796, within ±1 0.990 (test n=13716)

