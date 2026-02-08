# Jacobián & Inverzná kinematika — Pohyb robotického ramena

> **Kde sa to používa v našom kóde:**
> [`ik_controller.py`](../../src/mujoco_robot/core/ik_controller.py) — metóda `IKController.solve()`
> [`reach_env.py`](../../src/mujoco_robot/envs/reach_env.py) — volá `ik.solve()` každý riadiaci krok

> **Predpoklady:** [01 — 3D Rotácie & Kvaterniány](01_3d_rotacie_a_quaterniony.md)

> 🇬🇧 [English version](../02_jacobian_and_inverse_kinematics.md)

---

## Obsah

1. [Celkový obraz](#1-celkový-obraz)
2. [Priama kinematika (FK) — „Ak pohnem týmto kĺbom, kam sa dostane ruka?"](#2-priama-kinematika-fk)
3. [Jacobián — Prepojenie rýchlostí kĺbov a EE](#3-jacobián--prepojenie-rýchlostí-kĺbov-a-ee)
4. [Inverzná kinematika (IK) — „Ako pohnem kĺbmi, aby som sa dostal tam?"](#4-inverzná-kinematika-ik)
5. [Pseudo-inverzia — Prvé riešenie](#5-pseudo-inverzia--prvé-riešenie)
6. [Tlmené najmenšie štvorce (DLS) — Robustné riešenie](#6-tlmené-najmenšie-štvorce-dls--robustné-riešenie)
7. [Ako náš kód všetko spája](#7-ako-náš-kód-všetko-spája)
8. [Bežné úskalia & intuície](#8-bežné-úskalia--intuície)

---

## 1. Celkový obraz

Robotické rameno má **kĺby** (veci, ktoré rotujú) a **koncový efektor** (nástroj na špičke). Máme dve základné otázky:

| Otázka | Názov | Jednoduché? |
|--------|-------|-------------|
| „Ak nastavím každý uhol kĺbu na X, kde je ruka?" | **Priama kinematika (FK)** | ✅ Priamočiare |
| „Chcem ruku SEM — aké uhly kĺbov potrebujem?" | **Inverzná kinematika (IK)** | ❌ Ťažké! |

FK je ako tlačenie domina dopredu — je to jednoduchý reťazec výpočtov. IK je ako snaha zistiť, ktoré domino tlačiť, aby posledné pristálo na konkrétnom mieste — je to oveľa ťažšie a môže existovať **viacero riešení** alebo **žiadne riešenie**.

---

## 2. Priama kinematika (FK)

### Reťazec transformácií

Robotické rameno je reťazec tuhých článkov spojených kĺbmi. Každý kĺb sa otáča o nejaký uhol $q_i$. Na nájdenie polohy koncového efektora vynásobíme spolu všetky transformačné matice:

$$
T_{ee} = T_0 \cdot T_1(q_1) \cdot T_2(q_2) \cdot \ldots \cdot T_n(q_n)
$$

Každá $T_i$ je 4×4 **homogénna transformačná matica**, ktorá kóduje rotáciu aj transláciu:

$$
T = \begin{bmatrix} R_{3\times3} & \mathbf{p}_{3\times1} \\ \mathbf{0}_{1\times3} & 1 \end{bmatrix}
$$

kde $R$ je rotačná matica a $\mathbf{p}$ je translačný vektor.

### Čo nám FK dáva

Výsledok $T_{ee}$ nám hovorí:
- **Pozíciu**: translačná časť $\mathbf{p}_{ee}$ = (x, y, z) vo svete
- **Orientáciu**: rotačná časť $R_{ee}$ = 3×3 matica (ktorú prevedieme na kvaternión)

### MuJoCo robí FK za nás

FK nepočítame ručne — MuJoCo to robí každý simulačný krok. My len čítame výsledky:

```python
# Pozícia — MuJoCo ju vypočítal z uhlov kĺbov cez FK
pos = data.site_xpos[ee_site]  # (3,) pole

# Orientácia — 3×3 rotačná matica, tiež z FK
mat = data.site_xmat[ee_site].reshape(3, 3)
quat = _mat_to_quat(mat)       # prevod na kvaternión
```

---

## 3. Jacobián — Prepojenie rýchlostí kĺbov a EE

### Kľúčová otázka

Ak **mierne zmeníme** uhly kĺbov, ako sa pohne koncový efektor?

Toto je otázka o **deriváciách** — a odpoveďou je **Jakobiánova matica** (Jacobián).

### Definícia

Jacobián $J$ je matica, ktorá mapuje rýchlosti kĺbov $\dot{q}$ na rýchlosti koncového efektora $\dot{x}$:

$$
\dot{x} = J(q) \cdot \dot{q}
$$

kde:
- $\dot{q} \in \mathbb{R}^n$ = rýchlosti kĺbov (pre 6-kĺbové rameno, n=6)
- $\dot{x} \in \mathbb{R}^m$ = rýchlosť koncového efektora (u nás m=6: 3 lineárne + 3 uhlové)
- $J \in \mathbb{R}^{m \times n}$ = Jakobiánova matica

### Čo znamená každý riadok a stĺpec

Pre naše 6-DOF rameno s plným riadením pozície + orientácie:

$$
J = \begin{bmatrix}
\frac{\partial p_x}{\partial q_1} & \frac{\partial p_x}{\partial q_2} & \cdots & \frac{\partial p_x}{\partial q_6} \\[4pt]
\frac{\partial p_y}{\partial q_1} & \frac{\partial p_y}{\partial q_2} & \cdots & \frac{\partial p_y}{\partial q_6} \\[4pt]
\frac{\partial p_z}{\partial q_1} & \frac{\partial p_z}{\partial q_2} & \cdots & \frac{\partial p_z}{\partial q_6} \\[4pt]
\frac{\partial \omega_x}{\partial q_1} & \frac{\partial \omega_x}{\partial q_2} & \cdots & \frac{\partial \omega_x}{\partial q_6} \\[4pt]
\frac{\partial \omega_y}{\partial q_1} & \frac{\partial \omega_y}{\partial q_2} & \cdots & \frac{\partial \omega_y}{\partial q_6} \\[4pt]
\frac{\partial \omega_z}{\partial q_1} & \frac{\partial \omega_z}{\partial q_2} & \cdots & \frac{\partial \omega_z}{\partial q_6}
\end{bmatrix}
$$

- **Riadky 1-3** (translačný Jacobián $J_p$): Ako každý kĺb ovplyvňuje **pozíciu** EE
- **Riadky 4-6** (rotačný Jacobián $J_r$): Ako každý kĺb ovplyvňuje **uhlovú rýchlosť** EE
- **Každý stĺpec**: Vplyv jedného kĺbu na celú rýchlosť EE

### Fyzikálna intuícia: stĺpce ako „vektory vplyvu"

Stĺpec $j$ Jacobiánu je rýchlosť EE, ktorú by ste dostali, keby sa pohyboval **iba** kĺb $j$ jednotkovou rýchlosťou. Predstavte si ho ako „vplyv kĺbu $j$ na koncový efektor."

- **Základný kĺb** (rameno): otáčanie ním pohybuje EE vo veľkom oblúku → veľké položky v $J_p$
- **Kĺb zápästia**: sotva pohne pozíciou EE (malé položky v $J_p$), ale silno ho otáča (veľké položky v $J_r$)

### MuJoCo počíta Jacobián za nás

```python
jacp = np.zeros((3, model.nv))  # translačný Jacobián
jacr = np.zeros((3, model.nv))  # rotačný Jacobián
mujoco.mj_jacSite(model, data, jacp, jacr, ee_site)

# Poskladáme ich do plného 6×n Jacobiánu
J = np.vstack([jacp[:, robot_dofs], jacr[:, robot_dofs]])  # (6, 6)
```

Poznámka: `model.nv` je celkový počet rýchlostných DOF v modeli (môže zahŕňať voľne plávajúcu základňu, atď.), preto vyberieme len stĺpce kĺbov nášho robota pomocou `robot_dofs`.

---

## 4. Inverzná kinematika (IK)

### Problém

Chceme nájsť $\dot{q}$ tak, aby sa EE pohyboval smerom k **cieľovej póze**. Vieme:
- **Kde sme**: aktuálna póza EE (z FK)
- **Kde chceme byť**: cieľová pozícia + cieľový kvaternión
- **Jacobián**: $J$ v aktuálnej konfigurácii

**Chyba** medzi aktuálnou a cieľovou je:

$$
\mathbf{e} = \begin{bmatrix} \mathbf{p}_{ciel} - \mathbf{p}_{aktualna} \\ \text{chyba\_os\_uhol}(q_{aktualny}, q_{cielovy}) \end{bmatrix}
$$

Toto je 6-D vektor: 3 pre chybu pozície + 3 pre chybu orientácie (ako os-uhol, pozri predchádzajúci sprievodca).

### Ideálna rovnica

Chceme: $J \dot{q} = \mathbf{e}$

Ak je $J$ štvorcový (6×6) a nesingulárny: $\dot{q} = J^{-1} \mathbf{e}$

Ale sú tu problémy:
1. $J$ nemusí byť štvorcový (viac kĺbov ako je potrebné → redundancia, alebo menej → podaktuovanie)
2. $J$ môže byť **singulárny** (v určitých konfiguráciách sa niektoré smery stanú nedosiahnuteľné)
3. Priama inverzia je **numericky nestabilná** blízko singularít

---

## 5. Pseudo-inverzia — Prvé riešenie

### Riešenie najmenšími štvorcami

Keď $J$ nie je invertovateľný, hľadáme $\dot{q}$, ktoré minimalizuje $\|J\dot{q} - \mathbf{e}\|^2$ (najmenšie štvorce):

$$
\dot{q} = J^T (J J^T)^{-1} \mathbf{e}
$$

Toto je **pravá pseudo-inverzia** $J^\dagger = J^T (J J^T)^{-1}$.

Pre náš štvorcový 6×6 Jacobián to dáva rovnaký výsledok ako $J^{-1}$, keď je $J$ invertovateľný.

### Problém singularity 💥

Keď robot dosiahne určité konfigurácie (nazývané **singularity**), $J J^T$ sa stáva singulárnym (determinant → 0). Fyzikálne to znamená, že niektoré smery EE sa stávajú nedosiahnuteľnými — bez ohľadu na to, aké rýchlosti kĺbov aplikujete, EE sa nemôže pohybovať daným smerom.

**Príklad**: Plne vystrčené rameno. Nemôže sa pohybovať ďalej von — Jacobián nemá zložku v radiálnom smere. V tomto bode je $J J^T$ takmer singulárny a $(J J^T)^{-1}$ produkuje **obrovské** (alebo nekonečné) rýchlosti kĺbov.

To je zlé — robot by divoko roztáčal kĺby v snahe dosiahnuť nemožný pohyb.

---

## 6. Tlmené najmenšie štvorce (DLS) — Robustné riešenie

### Oprava: pridať tlmenie

Namiesto $(J J^T)^{-1}$ počítame:

$$
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}
$$

kde $\lambda$ je malý tlmiaci faktor (používame $\lambda = 0.02$).

### Čo tlmenie robí?

Člen $\lambda^2 I$ pridáva $\lambda^2$ k diagonále $J J^T$ **pred** inverziou. Toto:
- **Zabraňuje deleniu nulou** v singularitách
- **Obmedzuje maximálne rýchlosti kĺbov** — aj v singularitách zostáva $\dot{q}$ ohraničené
- **Vymieňa presnosť za robustnosť** — blízko singularít sa EE nepohne presne tam, kam chceme, ale robot sa nezbláznì

### Kompromis tlmenia

| $\lambda$ príliš malé | $\lambda$ príliš veľké |
|------------------------|------------------------|
| Takmer dokonalé sledovanie | Pomalé sledovanie |
| Divé kĺby v singularitách | Hladké kĺby všade |
| Numericky nestabilné | Veľmi stabilné |

Naša hodnota $\lambda = 0.02$ je dobrý kompromis — je dostatočne malá pre presné sledovanie, ale dostatočne veľká na zabránenie explóziám kĺbov blízko singularít.

### Optimalizačný pohľad

DLS minimalizuje kompromis medzi chybou sledovania a úsilím kĺbov:

$$
\min_{\dot{q}} \left( \|J\dot{q} - \mathbf{e}\|^2 + \lambda^2 \|\dot{q}\|^2 \right)
$$

Prvý člen chce presné sledovanie EE. Druhý penalizuje veľké rýchlosti kĺbov. $\lambda$ riadi rovnováhu.

---

## 7. Ako náš kód všetko spája

Tu je naša metóda `IKController.solve()`, anotovaná krok po kroku:

```python
def solve(self, target_pos, target_quat):
    # Krok 1: Získať Jacobián z MuJoCo
    jacp = np.zeros((3, self.model.nv))   # translačný (3 × nv)
    jacr = np.zeros((3, self.model.nv))   # rotačný    (3 × nv)
    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

    # Krok 2: Vypočítať 6-D chybový vektor
    pos_err = target_pos - self.data.site_xpos[self.ee_site]  # (3,)
    ori_err = orientation_error_axis_angle(self.ee_quat(), target_quat)  # (3,)
    target_vec = np.concatenate([pos_err, ori_err])  # (6,)

    # Krok 3: Zostaviť plný 6×n Jacobián (len kĺby nášho robota)
    cols = self.robot_dofs
    J = np.vstack([jacp[:, cols], jacr[:, cols]])  # (6, n_klbov)

    # Krok 4: Riešenie tlmenými najmenšími štvorcami
    lam = self.damping
    JJT = J @ J.T + (lam ** 2) * np.eye(6)   # (6, 6) — vždy invertovateľné!
    return J.T @ np.linalg.solve(JJT, target_vec)  # (n_klbov,)
```

### Čo sa deje v každom riadiacom kroku

```
    ┌─────────────────────────────────────────────────┐
    │               RIADIACA SLUČKA                    │
    │                                                  │
    │  1. RL politika vydá akciu  →  [dx,dy,dz,        │
    │                                 dwx,dwy,dwz]     │
    │                                                  │
    │  2. Akcia sa integruje do   →  ciel_poz,         │
    │     kartézskeho cieľa          ciel_quat          │
    │                                                  │
    │  3. IK regulátor vypočíta   →  rýchlosti kĺbov   │
    │     J^T(JJ^T + λ²I)⁻¹ chyba   (6 čísel)         │
    │                                                  │
    │  4. Rýchlosti kĺbov sa     →  MuJoCo ciele       │
    │     prevedú na pozičné         aktuátorov         │
    │     ciele: q_novy = q + dt*dq                    │
    │                                                  │
    │  5. MuJoCo simuluje fyziku  →  nové uhly kĺbov   │
    │     (PD riadenie + kontakty)   & póza EE          │
    │                                                  │
    │  6. Nové pozorovanie        →  späť na krok 1    │
    └─────────────────────────────────────────────────┘
```

---

## 8. Bežné úskalia & intuície

### Singularity — keď rameno „zasekne"

**Plne vystrčené rameno**: Nemôže sa pohybovať ďalej von. Riadok Jacobiánu pre radiálny pohyb sa stáva nulovým.

**Zložené rameno so zápästím zarovnaným s ramenom**: Dva kĺby sa stanú ekvivalentnými — oba rotujú okolo tej istej efektívnej osi. Jacobián stráca hodnosť (dva stĺpce sa stanú lineárne závislé).

DLS oba prípady rieši elegantne obetovaním dokonalého sledovania blízko singularít.

### Redundancia — keď je príliš veľa riešení

6-DOF rameno má presne 6 kĺbov a 6 DOF EE (3 pozícia + 3 orientácia). To znamená, že systém je **štvorcový** — typicky jedno riešenie.

7-DOF rameno (ako mnohé humanoidné ramená) má extra kĺb — je **redundantné**. Existuje nekonečne veľa konfigurácií kĺbov, ktoré dosiahnu tú istú pózu EE. Pseudo-inverzia dáva riešenie s „minimálnou normou" (najmenšie rýchlosti kĺbov), ale môžete pridať pohyby v **nulovom priestore** (pohyby kĺbov, ktoré vôbec neovplyvňujú EE) pre sekundárne ciele ako vyhýbanie sa limitom kĺbov.

### Prečo pozičné riadenie, nie rýchlostné?

Náš kód prevádza výstupy IK rýchlostí na **pozičné ciele**:

```python
q_ciel = q_aktualny + dt * q_rychlost_z_ik
```

Je to preto, že aktuátory MuJoCo sú **pozičné servá** — používajú interné PD regulátory na sledovanie pozičných cieľov. Takto fungujú aj skutočné UR roboty: posielate príkazy pozície kĺbov a vstavaný regulátor robota sa stará o nízkoúrovňové momenty sily.

### Mierka pozičných vs. orientačných chýb

Pozičné chyby sú v **metroch** a orientačné chyby sú v **radiánoch**. Tie majú rôzne fyzikálne mierky:
- 0.01m = 1cm (malá pozičná chyba)
- 0.01 rad ≈ 0.57° (veľmi malý uhol)

Ak sú mierky veľmi odlišné, IK sa môže sústrediť na jednu na úkor druhej. Naša funkcia odmeny to rieši použitím rôznych váh: $-0.2 \times \text{vzd}$ pre pozíciu a $-0.1 \times \text{chyba\_ori}$ pre orientáciu.

---

## Matematické zhrnutie

| Symbol | Význam | Veľkosť |
|--------|--------|---------|
| $q$ | Uhly kĺbov | $(n,)$ |
| $\dot{q}$ | Rýchlosti kĺbov | $(n,)$ |
| $\mathbf{p}$ | Pozícia EE | $(3,)$ |
| $\omega$ | Uhlová rýchlosť EE | $(3,)$ |
| $\mathbf{e}$ | 6-D chyba pózy (pozícia + orientácia) | $(6,)$ |
| $J$ | Plný Jacobián | $(6 \times n)$ |
| $J_p$ | Translačný Jacobián | $(3 \times n)$ |
| $J_r$ | Rotačný Jacobián | $(3 \times n)$ |
| $\lambda$ | DLS tlmiaci faktor | skalár |
| $I$ | Jednotková matica | $(6 \times 6)$ |

**Vzorec DLS:**

$$
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}
$$

---

**Predchádzajúce:** [01 — 3D Rotácie & Kvaterniány](01_3d_rotacie_a_quaterniony.md)
**Ďalej:** [03 — RL Prostredie](03_rl_prostredie.md) — ako formulujeme dosahovanie robota ako problém posilňovaného učenia.
