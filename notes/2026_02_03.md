# 2026.02.03.

```
identity_matrix = torch.eye(hi.shape[1], device=hi.device)
if hi.shape[0] > hi.shape[1]:
    if self.rcond is not None:
        weights2.data = (
                torch.linalg.pinv(hi.T @ hi + identity_matrix / self.penalty_term, rcond=self.rcond)
                @ (hi.T @ train_y)
        )
    else:
        weights2.data = (
                torch.linalg.pinv(hi.T @ hi + identity_matrix / self.penalty_term) @ (hi.T @ train_y)
        )
else:
    if self.rcond is not None:
        weights2.data = (
                hi.T @ torch.linalg.pinv(hi @ hi.T + identity_matrix / self.penalty_term,
                                            rcond=self.rcond) @ train_y
        )
    else:
        weights2.data = (
                hi.T @ torch.linalg.pinv(hi @ hi.T + identity_matrix / self.penalty_term) @ train_y
        )
```

Ha a minták száma ($N$) nagyobb, mint a jellemzőké ($L$)

1. Pszeudoinverz bemeneti mátrix kiszámítása:
    
    ```
    psuedo_inv_input = torch.linalg.pinv(
	    hi.T @ hi + identity_matrix / self.penalty_term
	)
    ```
    $$pseudo_{inv\_input} = \mathbf{H}^\top \mathbf{H} + \frac{\mathbf{I}}{C}$$
    
    _Dimenziókkal behelyettesítve:_
    
    $$(2049, 47290) \times (47290, 2049) + (2049, 2049) = \mathbf{(2049, 2049)}$$
    
2. **Súlyok meghatározása:**
    ```
    weights2 = psuedo_inv_input @ (hi.T @ train_y)
    ```
    $$\mathbf{W}_2 = (pseudo_{inv\_input})^{-1} \times (\mathbf{H}^\top \mathbf{Y})$$
    
    _Dimenziókkal:_
    
    $$(2049, 2049) \times ((2049, 47290) \times (47290, 3)) = \mathbf{(2049, 3)}$$


## Új algoritmus

```
def _get_class_stats(self, hidden_layer):  
    _, y_one_hot = next(iter(self.train_loader))  
    y_true = torch.argmax(y_one_hot, dim=-1)  
    y_pred = torch.argmax(self.predictions, dim=-1)  
    num_classes = y_one_hot.size(1)  
```

Kinyerem a tanító adatok közül a GT címkéket, valamint a hálózat által predikáltakat.

```
    # class means in the hidden layer  
    classes = torch.unique(y_true).sort()[0]  
    means = torch.stack([hidden_layer[y_true == c].mean(dim=0) for c in classes])  

    # direction vectors and Euc. dist.  
    direction_tensor = means.unsqueeze(1) - means.unsqueeze(0)  
    dist_matrix = torch.norm(direction_tensor, dim=-1, p=2)  
```
Kiszámítom az osztályok átlagvektorait a rejtett rétegben, amiből előáll az osztálypárok közötti **irányvektorok** és **távolságmátrix**.

```  
    # confusion matrix  
    conf = torch.zeros((num_classes, num_classes))  
    for t, p in zip(y_true, y_pred):  
        conf[t, p] += 1  
  
    # Normalized error matrix  
    error_matrix = conf / (conf.sum(dim=1, keepdim=True) + 1e-8)  
  
    return direction_tensor, dist_matrix, error_matrix
```
Kiszámolom a **normalizált keveredési mátrixot**, amely megmutatja az osztályok közötti tévesztési arányokat.

```
def _allocate_neurons_per_class_pair(self, hidden_layer, total_neurons: int, eps: float = 1e-8):  
    dir_tensor, dist_mtx, err_mtx = self._get_class_stats(hidden_layer)  
  
    difficulty_score = (err_mtx + err_mtx.T) / (dist_mtx + eps)  
  
    C = difficulty_score.shape[0]  
    triu_idx = torch.triu_indices(C, C, offset=1)  
    scores = difficulty_score[triu_idx[0], triu_idx[1]]  
```
Első lépésben a  meghatározom, melyik osztálypár minősül nehéz esetnek. Egy pontszámot rendelek hozzájuk: minél többször keveri őket össze a hálózat (hiba) és minél közelebb vannak egymáshoz (távolság), annál magasabb ez a pontszám.

```
    # Sorting, most difficult ones first  
    sorted_indices = torch.argsort(scores, descending=True)  
```
Az osztálypárokat a nehézségi pontszám alapján sorba rendezem.

```
    num_pairs = len(sorted_indices)  
    allocation = {}  
  
    # Allocation  
    for i in range(num_pairs):  
        c1, c2 = triu_idx[0][i].item(), triu_idx[1][i].item()  
        allocation[(int(c1), int(c2))] =1  
  
    remaining = total_neurons - num_pairs  
    if remaining > 0:  
        weights = scores / (scores.sum() + eps)  
        extra_neurons = torch.floor(weights * remaining).int()  
  
        for i, idx in enumerate(sorted_indices):  
            c1, c2 = triu_idx[0][idx].item(), triu_idx[1][idx].item()  
            allocation[(int(c1), int(c2))] += extra_neurons[idx].item()  
```

Elkezdem kiosztani a foglalást. Minden osztálypár kap legalább egy neuront, a maradék szabad helyeket (neuronokat) a nehézségi pontszámok arányában osztom szét. Minél nehezebb egy pár szétválasztása, annál több új neuront kap majd.

```
# Fix rounding  
    diff = total_neurons - sum(allocation.values())  
    for i in range(abs(diff)):  
        idx = sorted_indices[i % num_pairs]  
        pair = int(triu_idx[0][idx]), int(triu_idx[1][idx])  
        allocation[pair] += 1 if diff > 0 else -1  
  
    return allocation, dir_tensor
```

Itt csak a kerekítési hibákat javítom.

```
def _create_hidden_layer(self, weights: torch.Tensor, eps: float = 1e-8):  
    dimension, _ = weights.shape  
  
    noise = torch.normal(mean=self.mu, std=self.sigma, size=weights.shape)  
    w_rnd_out_i = weights + noise  
```
Az előző rétegből megtartom a kimeneti súlyokat, ezeket lemásolom és zajt adok hozzájuk.

```  
    new_columns = []  
  
    for (c1, c2), n_neurons in self.allocation.items():  
        # base direction from class means  
        v_base = self.class_direction_tensor[c1, c2]  
        v_unit = v_base / (v_base.norm() + eps)  
  
        for i in range(n_neurons):  
            if i == 0:  
                v = v_unit  
            else:  
                v_noise = torch.normal(mean=0.0, std=self.sigma, size=(dimension,))  
                v = v_unit + v_noise  
                v = v / (v.norm() + eps)  
  
            new_columns.append(v.view(dimension, 1))  
  
    if not new_columns:  
        raise ValueError("new_columns is empty!")  
```
Minden problémás párhoz készítek egy tiszta, módosítatlan irányvektort, a többit enyhe zajjal módosítva adom hozzá. (Az irányvektorok az előző rejtett rétegben kiszámolt osztályátlagok  közötti különbségvektorok.)

```  
    hidden_layer = torch.cat([weights, w_rnd_out_i, torch.cat(new_columns, dim=1)], dim=1)  
  
    return hidden_layer
```

Itt a végén pedig csak összefűzöm egy réteggé a három építő elemet.

## Megjegyzések, észrevételek

- Alapvető problémánk, hogy a 3. rétegben elkezd csökkeni a pontosság (jobb esetben csak a teszt).
- Visszaírtam az eredeti _create_hidden_layer függvényt (ahol ortogonális súlyokat készítettünk), úgy, hogy bent hagytam az eddig megírt, új függvényeket. Az történt ami várható volt; úgy működött minden mint eddig.
- Próbálkoztam azzal, hogy a sigma minden réteg esetén csökkenjen;
```
noise = torch.normal(mean=self.mu, std=self.sigma, size=weights.shape)
```
	illetve
```
v_noise = torch.normal(mean=0.0, std=self.sigma, size=(dimension,))
```
	de alapvetően nem igazán hozott változást.
- Gondoltam arra, hogy kiszámolom a súlymátrixok kondícióját. Három évvel ezelőtt, mikor tovább akartuk fejleszteni a módszert, egy ilyen problémába belefutottunk; instabilak lettek a mátrixok, ezért elkezdett csökkeni a pontosság. 
	- Jelen esetben ez nem áll fent. Bár lehet, hogy valami plotot rajzolnom kéne.
- Volt egy ilyen ötletem, hogy mi lenne, ha a második rétegben alkalmaznám az új megközelítést, míg a harmadikban vissza állnék az eredetire (tehát ortogonális random súlyok). 
	- Ekkor megszűnt a probléma, nem csökkent, hanem nőtt a pontosság.
- Ellenőriztem minden függvényhívást, paraméter átadást, stb; minden rendben van, akkor és az hívódik meg aminek és amikor kell.

Bemásolok ide egy példa futtatást, most ezzel a legutóbbi megoldással (második réteg új módszer, harmadik a régivel):
(A hosszabb futásidő azért van mert az itthoni laptopomról dolgozom)
```
INFO     JSON data is valid.
INFO     Config DataFrame:
                             Value
dataset_name                letter
activation               LeakyReLU
number_of_tests                  1
seed                          True
mu                               0
sigma                          0.1
exp_neurons      [3000, 1000, 500]
penalty                         11
rcond                          0.0
INFO     Size of train dataset: 14000, Size of test dataset: 3000
Process:   0%|          | 0/1 [00:00<?, ?it/s]
Training:   0%|          | 0/1 [00:00<?, ?it/s]
Training: 100%|██████████| 1/1 [00:07<00:00,  7.50s/it]
INFO     Execution time of train_ith_layer: 7.5030 seconds
INFO     train accuracy: 0.9638
INFO     train precision: 0.9644
INFO     train recall: 0.9635
INFO     train F1-score: 0.9637
INFO     test accuracy: 0.9373
INFO     test precision: 0.9385
INFO     test recall: 0.9366
INFO     test F1-score: 0.9367

Training:   0%|          | 0/1 [00:00<?, ?it/s]
Training: 100%|██████████| 1/1 [00:01<00:00,  1.76s/it]
INFO     Execution time of train_ith_layer: 1.7599 seconds
INFO     train accuracy: 0.9714
INFO     train precision: 0.9717
INFO     train recall: 0.9712
INFO     train F1-score: 0.9713
INFO     test accuracy: 0.9433
INFO     test precision: 0.9433
INFO     test recall: 0.9425
INFO     test F1-score: 0.9423

Training:   0%|          | 0/1 [00:00<?, ?it/s]
Training: 100%|██████████| 1/1 [00:00<00:00,  2.67it/s]
INFO     Execution time of train_ith_layer: 0.3739 seconds
INFO     train accuracy: 0.9726
INFO     train precision: 0.9729
INFO     train recall: 0.9725
INFO     train F1-score: 0.9726
INFO     test accuracy: 0.9463
INFO     test precision: 0.9466
INFO     test recall: 0.9455
INFO     test F1-score: 0.9455
Process: 100%|██████████| 1/1 [00:14<00:00, 14.68s/it]
```