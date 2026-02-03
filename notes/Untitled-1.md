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

# A képlet és a kód összhangban van egymással? Az lett implementálva ami a képletben szerepel?

Igen, a kódod matematikailag összhangban van a képpel, de van egy apró implementációs különbség a regularizációs tag ($C$) kezelésében, amire érdemes figyelned. Nézzük meg részletesen:

A képlet vs. kód elemzése

A képen látható képlet az Extreme Learning Machine (ELM) regularizált pszeudoinverzét ($\mathbf{H}^{\dagger}$) határozza meg két esetre:

Ha $N > L$ (több adat, mint neuron): $$\mathbf{H}^{\dagger} = (C^{-1}\mathbf{I} + \mathbf{H}^{\top}\mathbf{H})^{-1}\mathbf{H}^{\top}$$

A kódodban: torch.linalg.pinv(hi.T @ hi + identity_matrix / self.penalty_term) @ (hi.T @ train_y)Megfeleltetés: Itt a self.penalty_term felel meg a $C$ paraméternek. Mivel a képletben $C^{-1}$ szerepel, a kódban használt 1 / self.penalty_term matematikailag pontosan ugyanaz.

Ha $N < L$ (több neuron, mint adat):$$\mathbf{H}^{\dagger} = \mathbf{H}^{\top}(C^{-1}\mathbf{I} + \mathbf{H}\mathbf{H}^{\top})^{-1}$$A kódodban: hi.T @ torch.linalg.pinv(hi @ hi.T + identity_matrix / self.penalty_term) @ train_y

Megfeleltetés: Ez is pontosan követi a képlet szerkezetét.

Összegzés Igen, az lett implementálva, ami a képletben szerepel. A kódod hűen követi mind a dimenzióváltás (N vs L), mind a regularizációs tag elhelyezésének szabályait.

# Jól értelmezem hogy nem a teljes méretű H-val számolunk?

Igen, pontosan jól értelmezed, és ez az ELM (Extreme Learning Machine) egyik legfontosabb matematikai trükkje a hatékonyság érdekében.A kódodban és a képletben is látható szétválasztás ($N > L$ vagy $N < L$) azt szolgálja, hogy mindig a kisebb dimenziójú mátrixot kelljen invertálni.

Miért nem a "teljes" $\mathbf{H}$-val számolunk?

A $\mathbf{H}$ mátrix mérete $N \times L$, ahol:$N$: a tanító adatok száma (minták).$L$: a rejtett neuronok száma (hidden nodes).
Ha közvetlenül a $\mathbf{H}$ pszeudoinverzét akarnád kiszámolni, az egy $N \times L$ méretű művelet lenne, ami hatalmas memóriaigénnyel járna. Ehelyett a kódod a következő két négyzetes mátrix egyikét használja:
Ha $N > L$ (Sok adat, kevés neuron):

A $\mathbf{H}^\top \mathbf{H}$ mátrixszal számolsz.Ennek mérete $L \times L$.
Ez azért jó, mert ha van 1 millió adatod ($N$), de csak 1000 neuronod ($L$), akkor egy $1000 \times 1000$-es mátrixot sokkal könnyebb invertálni, mint az 1 milliósat.

Ha $N < L$ (Kevés adat, sok neuron):A $\mathbf{H} \mathbf{H}^\top$ mátrixszal számolsz.

Ennek mérete $N \times N$.
Ez akkor fordul elő, ha extrém sok neuront használsz kis adathalmazon.

Itt a minták száma határozza meg a számítási költséget.

Összegezve a kódban: Az if hi.shape[0] > hi.shape[1]: feltétel pontosan ezt ellenőrzi:hi.shape[0]: ez az $N$ (sorok száma).hi.shape[1]: ez az $L$ (oszlopok/neuronok száma).

A válasz tehát: Igen. Nem a teljes $N \times L$ mátrixot invertálod közvetlenül, hanem a transzponálttal való visszaszorzás után kapott kisebb, négyzetes mátrixot, így spórolsz rengeteg memóriát és számítási időt. Ez teszi lehetővé, hogy az ELM nagyságrendekkel gyorsabb legyen, mint a hagyományos backpropagation.

# De így nem számol az összes bemeneti adattal, nem?

Ez egy nagyon gyakori és jogos kérdés, amikor először látja valaki ezt a matematikai trükköt! A válasz az, hogy de, a hálózat minden egyes bemeneti adatot figyelembe vesz, egyetlen mintát sem hagy ki a számításból.

A titok abban rejlik, hogy a mátrixszorzás ($\mathbf{H}^\top \mathbf{H}$ vagy $\mathbf{H} \mathbf{H}^\top$) "összetömöríti" az információt, de nem vész el belőle semmi, ami a súlyok kiszámításához kell.Hogyan marad benne az összes adat?

Gondolj a $\mathbf{H}^\top \mathbf{H}$ műveletre úgy, mint egy összegzésre:

A teljes mátrix: $\mathbf{H}$ tartalmazza az összes $N$ darab adatsort. A szorzás eredménye: Amikor a $\mathbf{H}^\top$-t megszorzod $\mathbf{H}$-val, valójában minden egyes adatsornak kiszámolod az önmagával vett szorzatát, és ezeket a szorzatokat összeadod. 

Az eredmény: A kapott $L \times L$ mátrix minden egyes eleme az összes $N$ darab minta hozzájárulását tartalmazza.Egy egyszerű példa (Szemléltetés)Képzeld el, hogy az átlagot akarod kiszámolni 1 millió számból ($N=1.000.000$). 

A "teljes" módszer: Minden számot elmentesz egy listába, majd a végén összeadod őket.

Az "összegzős" módszer: 
Csak egyetlen változót tartasz fent (sum), és minden beérkező számot hozzáadsz.A végén mindkét módszerrel ugyanazt az átlagot kapod. Az ELM mátrixművelete is ilyen: a $\mathbf{H}^\top \mathbf{H}$ egy olyan "összegző mátrix" (úgynevezett Gram-mátrix), ami sűrítve tartalmazza az összes adat statisztikai jellemzőjét.Miért nem baj, hogy a mátrix kisebb lesz?

Az ELM célja, hogy megtalálja az optimális kimeneti súlyokat ($\beta$). Ehhez nem kell minden adatot külön-külön "látnia" az invertálás pillanatában, csak azt kell tudnia, hogy az adatok összessége milyen összefüggéseket (korrelációkat) mutat.

Összegezve:
A bemeneti rétegen és a rejtett rétegen minden adat átmegy.A $\mathbf{H}$ mátrixba minden adat bekerül.A szorzás során az összes adat információja összegződik a kisebb négyzetes mátrixba.Az invertálás már ezen az "összesített tudáson" alapul.Tehát semmilyen adatvesztés nem történik, a matematikai végeredmény (a kiszámolt súlyok értéke) pontosan ugyanaz lesz, mintha egy elképzelhetetlenül nagy mátrixszal számolnál.
