# 3D Rotácie & Kvaterniány — Od Nuly po Hrdinu

> **Kde sa to používa v našom kóde:**
> [`ik_controller.py`](../../src/mujoco_robot/core/ik_controller.py) — všetky pomocné funkcie pre kvaterniány
> [`reach_env.py`](../../src/mujoco_robot/envs/reach_env.py) — cieľová orientácia, orientácia EE, chyba orientácie

> 🇬🇧 [English version](../01_3d_rotations_and_quaternions.md)

---

## Obsah

1. [Prečo potrebujeme rotácie?](#1-prečo-potrebujeme-rotácie)
2. [2D Rotácia — Intuitívny začiatok](#2-2d-rotácia--intuitívny-začiatok)
3. [3D Rotačné matice](#3-3d-rotačné-matice)
4. [Eulerove uhly — Intuitívny (ale problematický) spôsob](#4-eulerove-uhly--intuitívny-ale-problematický-spôsob)
5. [Kvaterniány — Robustný spôsob](#5-kvaterniány--robustný-spôsob)
6. [Reprezentácia os-uhol](#6-reprezentácia-os-uhol)
7. [Prevody medzi reprezentáciami](#7-prevody-medzi-reprezentáciami)
8. [Ako náš kód toto všetko využíva](#8-ako-náš-kód-toto-všetko-využíva)

---

## 1. Prečo potrebujeme rotácie?

Robotické rameno musí dosiahnuť **pózu** — to je pozícia (kde) plus orientácia (akým smerom ukazuje). Pozícia je jednoduchá: len 3 čísla (x, y, z). Ale orientácia je zložitejšia.

Predstavte si, že držíte skrutkovač. Môžete:
- **Namierenia** ho ľubovoľným smerom (to sú 2 stupne voľnosti — ako zemepisná šírka a dĺžka)
- **Otočiť** ho okolo jeho vlastnej osi (to je ďalší 1 stupeň voľnosti)

To sú **3 stupne voľnosti** (DOF) pre orientáciu. Ale reprezentovať tieto 3 DOF bez problémov sa ukazuje byť prekvapivo ťažké.

---

## 2. 2D Rotácia — Intuitívny začiatok

Pred skokom do 3D si vybudujme intuíciu v 2D.

### Jeden uhol

V 2D je rotácia jednoduchá — jeden uhol θ:

```
Otočiť bod (x, y) o uhol θ:

    x' = x·cos(θ) - y·sin(θ)
    y' = x·sin(θ) + y·cos(θ)
```

### Ako matica

Toto môžeme zapísať ako **rotačnú maticu**:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

Takže rotácia vektora $\mathbf{v}$ je len násobenie maticou: $\mathbf{v'} = R \cdot \mathbf{v}$.

**Kľúčová vlastnosť:** $R^T = R^{-1}$ (transpozícia JE inverzná matica). Toto robí maticu *rotačnou* maticou — je **ortogonálna**.

---

## 3. 3D Rotačné matice

### Základné rotácie okolo každej osi

V 3D môžeme rotovať okolo ktorejkoľvek z troch súradnicových osí:

**Rotácia okolo osi X** (náklon — roll):

$$
R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}
$$

**Rotácia okolo osi Y** (sklon — pitch):

$$
R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}
$$

**Rotácia okolo osi Z** (zatáčanie — yaw):

$$
R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

### Vlastnosti 3×3 rotačných matíc

Platná rotačná matica $R$ spĺňa:
- $R^T R = I$ (ortogonálna — stĺpce sú jednotkové vektory a navzájom kolmé)
- $\det(R) = +1$ (vlastná rotácia — žiadne zrkadlenie)
- **Stĺpce** $R$ sú otočené súradnicové osi

### Čo stĺpce fyzikálne znamenajú

Ak máte rotačnú maticu pre koncový efektor robota:

$$
R = \begin{bmatrix} | & | & | \\ \mathbf{x}_{ee} & \mathbf{y}_{ee} & \mathbf{z}_{ee} \\ | & | & | \end{bmatrix}
$$

- **Stĺpec 1** ($\mathbf{x}_{ee}$) = kam ukazuje os X koncového efektora (červená) vo svete
- **Stĺpec 2** ($\mathbf{y}_{ee}$) = kam ukazuje os Y koncového efektora (zelená) vo svete
- **Stĺpec 3** ($\mathbf{z}_{ee}$) = kam ukazuje os Z koncového efektora (modrá) vo svete

Presne toto zobrazujú **RGB osi** v našej MuJoCo vizualizácii!

### Skladanie rotácií

Na aplikáciu rotácie $R_1$ najprv, potom $R_2$:

$$
R_{celkova} = R_2 \cdot R_1
$$

⚠️ **Na poradí záleží!** $R_2 \cdot R_1 \neq R_1 \cdot R_2$ vo všeobecnosti. Skúste otočiť knihu: 90° okolo X a potom 90° okolo Z dáva iný výsledok ako Z-potom-X.

---

## 4. Eulerove uhly — Intuitívny (ale problematický) spôsob

Eulerove uhly popisujú rotáciu ako tri postupné rotácie okolo súradnicových osí. Napríklad **konvencia ZYX** (zatáčanie-sklon-náklon):

$$
R = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)
$$

kde $\psi$ = zatáčanie (yaw), $\theta$ = sklon (pitch), $\phi$ = náklon (roll).

### Prečo sú Eulerove uhly lákavé

- **Ľahko si ich predstaviť**: „otoč 30° doľava, nakloň 15° dopredu"
- **Kompaktné**: len 3 čísla
- **Ľudsky prívetivé**: piloti a herní vývojári ich používajú denne

### Prečo sú Eulerove uhly NEBEZPEČNÉ

#### Gimbal Lock (Zamknutie kardanového závesu) 🔒

Keď je stredná rotácia ±90°, stratíte jeden stupeň voľnosti. Prvá a tretia rotácia sa stanú ekvivalentnými — rotujú okolo tej istej osi.

**Príklad**: V konvencii ZYX, ak sklon = 90°, potom zatáčanie a náklon rotujú okolo tej istej osi. Nedokážete rozlíšiť zatáčanie od náklonu!

Toto nie je len matematická kuriozita — spôsobuje reálne problémy:
- **Interpolácia zlyhá** blízko gimbal lock-u
- **Riadenie sa stáva singulárnym** — Jacobián stráca hodnosť
- **Numerická nestabilita** blízko ±90° sklonu

#### Diskontinuity

Eulerove uhly sa "pretáčajú" (napr. 359° a 1° sú si blízke, ale numericky ďaleko). To robí výpočet „ako ďaleko sú dve orientácie?" nespoľahlivým.

### 💡 Preto sme prešli z yaw-only na kvaterniány

Náš skorší kód riadil len zatáčanie (jeden Eulerov uhol). To fungovalo dobre, lebo sme sa vyhýbali gimbal lock-u používaním len jednej rotácie. Ale na riadenie **všetkých 3 osí** potrebujeme kvaterniány.

---

## 5. Kvaterniány — Robustný spôsob

### Čo JE to kvaternión?

Kvaternión je 4-číselná reprezentácia 3D rotácie:

$$
q = w + xi + yj + zk = (w, x, y, z)
$$

kde:
- $w$ je **skalárna** (reálna) časť
- $(x, y, z)$ je **vektorová** (imaginárna) časť
- $i, j, k$ sú imaginárne jednotky so špeciálnymi pravidlami násobenia

### Geometrický význam

Jednotkový kvaternión $(w, x, y, z)$ predstavuje rotáciu o uhol $\theta$ okolo osi $\hat{n} = (n_x, n_y, n_z)$:

$$
q = \left(\cos\frac{\theta}{2},\;\; n_x\sin\frac{\theta}{2},\;\; n_y\sin\frac{\theta}{2},\;\; n_z\sin\frac{\theta}{2}\right)
$$

**Príklady:**
- **Žiadna rotácia** (identita): $q = (1, 0, 0, 0)$ → $\theta = 0$
- **90° okolo Z**: $q = (\cos 45°, 0, 0, \sin 45°) = (0.707, 0, 0, 0.707)$
- **180° okolo X**: $q = (\cos 90°, \sin 90°, 0, 0) = (0, 1, 0, 0)$

### Prečo polovičný uhol?

Faktor $\frac{\theta}{2}$ je to, čo zabezpečuje správne fungovanie kvaterniánovej algebry. Nie je ľubovoľný — vyplýva z matematickej štruktúry rotačnej grupy SO(3).

### Podmienka jednotkového kvaternióna

**Rotačný** kvaternión musí mať jednotkovú normu:

$$
\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1
$$

Túto podmienku náš kód vynucuje (pozri `_mat_to_quat`, ktorá na konci normalizuje).

### Násobenie kvaternionov (Hamiltonov súčin)

Na zloženie dvoch rotácií vynásobíme ich kvaterniány:

$$
q_1 \otimes q_2 = \begin{pmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{pmatrix}
$$

Toto je implementované v našej funkcii `quat_multiply()`.

⚠️ **Na poradí záleží!** $q_1 \otimes q_2 \neq q_2 \otimes q_1$ (rovnako ako pri rotačných maticiach).

### Konjugovaný kvaternión (inverzná rotácia)

Pre jednotkový kvaternión je **konjugát** inverzná rotácia:

$$
q^* = (w, -x, -y, -z)
$$

Toto je `quat_conjugate()` v našom kóde. Ak $q$ rotuje 90° v smere hodinových ručičiek okolo Z, potom $q^*$ rotuje 90° proti smeru hodinových ručičiek okolo Z.

### Problém dvojitého pokrytia: q a -q

Tu je jemný, ale kritický bod: **$q$ a $-q$ predstavujú tú istú rotáciu!**

$$
(w, x, y, z) \text{ a } (-w, -x, -y, -z) \text{ produkujú identické rotácie}
$$

Je to preto, že rotačný vzorec zahŕňa $q \mathbf{v} q^*$ a dvojitá negácia sa vyruší.

To vytvára problémy pri výpočte chýb (vzdialenosť medzi dvoma kvaternionmi môže ísť krátkou alebo dlhou cestou). Naša funkcia `quat_unique()` to rieši vynútením $w \geq 0$:

```python
def quat_unique(q):
    """Zabezpeč w ≥ 0 (vyriešenie nejednoznačnosti q / -q)."""
    return -q if q[0] < 0 else q.copy()
```

### Konvencia: WXYZ vs XYZW

⚠️ Rôzne knižnice používajú rôzne poradie:
- **MuJoCo, náš kód**: $(w, x, y, z)$ — skalár prvý
- **PyTorch3D, SciPy**: $(x, y, z, w)$ — skalár posledný

Vždy skontrolujte, akú konvenciu knižnica používa!

### Prečo sú kvaterniány lepšie ako Eulerove uhly

| Vlastnosť | Eulerove uhly | Kvaterniány |
|-----------|--------------|-------------|
| Parametre | 3 | 4 |
| Gimbal lock? | ÁNO ❌ | NIE ✅ |
| Plynulá interpolácia? | NIE ❌ | ÁNO ✅ (SLERP) |
| Skladanie | Zložitá trigonometria | Jednoduché násobenie |
| Výpočet chyby | Problémy s pretáčaním | Čistý os-uhol |
| Numerická stabilita | Slabá blízko singularít | Vynikajúca ✅ |

---

## 6. Reprezentácia os-uhol

### Čo je os-uhol?

Akákoľvek 3D rotácia sa dá opísať ako rotácia o uhol $\theta$ okolo jednotkovej osi $\hat{n}$:

$$
\text{vektor os-uhol} = \theta \cdot \hat{n} = (\theta n_x, \theta n_y, \theta n_z)
$$

Toto je **3-D vektor** kde:
- **Smer** = os rotácie
- **Veľkosť** = uhol rotácie (v radiánoch)

**Príklad**: Rotácia o 90° okolo osi Z → os-uhol = $(0, 0, \frac{\pi}{2})$

### Prevod kvaternión → os-uhol

Toto je `axis_angle_from_quat()` v našom kóde:

```python
def axis_angle_from_quat(q):
    q = quat_unique(q)            # zabezpeč w ≥ 0
    sin_half = ||q[1:4]||         # veľkosť vektorovej časti
    half_angle = atan2(sin_half, q[0])
    axis = q[1:4] / sin_half      # jednotková os rotácie
    return axis * (2 * half_angle) # uhol × os
```

Matematika:
- Keďže $q = (\cos\frac{\theta}{2}, \hat{n}\sin\frac{\theta}{2})$
- Norma vektorovej časti je $\sin\frac{\theta}{2}$
- Polovičný uhol je $\frac{\theta}{2} = \text{atan2}(\sin\frac{\theta}{2}, \cos\frac{\theta}{2})$
- Os je normalizovaná vektorová časť

### Prevod os-uhol → kvaternión

Daný vektor os-uhol $\mathbf{a} = \theta \hat{n}$:

$$
q = \left(\cos\frac{\theta}{2},\;\; \hat{n}\sin\frac{\theta}{2}\right)
$$

Toto sa používa v `_desired_ee()` na integráciu prírastkov orientácie:

```python
angle = np.linalg.norm(delta_ori)    # θ
axis = delta_ori / angle             # n̂
half = angle / 2.0
dq = [cos(half), axis * sin(half)]   # kvaternión z os-uhol
```

### Prečo je os-uhol skvelý pre chyby

**Chyba orientácie** medzi aktuálnou a cieľovou orientáciou sa prirodzene vyjadrí ako vektor os-uhol:

$$
\mathbf{e}_{ori} = \text{os\_uhol}(q_{ciel} \otimes q_{aktuálny}^*)
$$

Toto nám dáva:
- **3-D vektor**, ktorý môžeme poskytnúť IK regulátoru
- Jeho **veľkosť** je uhlová chyba v radiánoch
- Jeho **smer** nám hovorí, AKÝM SMEROM rotovať

Presne toto vypočítava `orientation_error_axis_angle()`:

```python
def orientation_error_axis_angle(current_quat, target_quat):
    q_err = quat_multiply(target_quat, quat_conjugate(current_quat))
    return axis_angle_from_quat(q_err)
```

---

## 7. Prevody medzi reprezentáciami

### Rotačná matica → Kvaternión (Shepperdova metóda)

Toto je `_mat_to_quat()` v našom kóde. Je to zložitejšie, než by ste čakali, pretože naivné vzorce majú numerické problémy.

Myšlienka: z rotačnej matice $R$ môžeme extrahovať:

$$
w = \frac{1}{2}\sqrt{1 + R_{00} + R_{11} + R_{22}}
$$

Ale keď je stopa $(R_{00} + R_{11} + R_{22})$ záporná, zahŕňa to odmocninu zo záporného čísla. **Shepperdova metóda** kontroluje, ktorý diagonálny element je najväčší a používa numericky stabilný vzorec pre každý prípad.

### Kvaternión → Rotačná matica

Daný $q = (w, x, y, z)$:

$$
R = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

### Zhrnutie všetkých reprezentácií

| Reprezentácia | Počet čísel | Výhody | Nevýhody |
|--------------|-------------|--------|----------|
| Rotačná matica | 9 (3×3) | Skladanie = násobenie, stĺpce = osi | Redundantná (6 podmienok) |
| Eulerove uhly | 3 | Ľudsky čitateľné | Gimbal lock, diskontinuita |
| Kvaternión | 4 | Žiadny gimbal lock, plynulý, rýchly | Dvojité pokrytie (q = -q), menej intuitívny |
| Os-uhol | 3 | Minimálny, fyzikálny význam | Singulárny pri θ=0 (os nedefinovaná) |

---

## 8. Ako náš kód toto všetko využíva

### Celý pipeline

1. **MuJoCo nám dáva** 3×3 rotačnú maticu pre EE site (`data.site_xmat`)

2. **My prevedieme** túto maticu → kvaternión pomocou `_mat_to_quat()` (Shepperdova metóda)

3. **Cieľová orientácia** je uložená ako kvaternión (`goal_quat`), vzorkovaný rovnomerne pomocou Shoemakeovej metódy

4. **Chyba orientácie** sa vypočíta ako vektor os-uhol:
   - $q_{chyba} = q_{ciel} \otimes q_{ee}^*$
   - $\mathbf{e}_{ori} = \text{axis\_angle\_from\_quat}(q_{chyba})$
   - Tento 3-D vektor je súčasne **pozorovanie** pre RL politiku aj súčasťou **IK cieľa**

5. **Veľkosť chyby** (skalár) sa používa v odmene:
   - $\text{ori\_chyba} = \|\mathbf{e}_{ori}\|$ (v radiánoch, 0 až π)
   - Príspevok k odmene: $-0.1 \times \text{ori\_chyba}$

6. **IK regulátor** používa 3-D vektor os-uhol chyby ako rotačnú zložku svojho 6-D cieľového vektora (pozri [sprievodcu Jacobiánom & IK](02_jacobian_a_inverzna_kinematika.md))

### Rovnomerné náhodné vzorkovanie kvaternionov (Shoemakeova metóda)

Keď vzorkujeme náhodnú cieľovú orientáciu, potrebujeme **rovnomerne rozdelenú** cez všetky možné rotácie. Jednoduché náhodné vygenerovanie 4 čísel a normalizácia NEDÁVA rovnomerné rotácie (zhluky sa tvoria pri póloch).

**Shoemakeova metóda** používa 3 rovnomerne rozdelené náhodné čísla $(u_1, u_2, u_3) \in [0,1)$:

$$
q = \begin{pmatrix}
\sqrt{1-u_1}\sin(2\pi u_2) \\
\sqrt{1-u_1}\cos(2\pi u_2) \\
\sqrt{u_1}\sin(2\pi u_3) \\
\sqrt{u_1}\cos(2\pi u_3)
\end{pmatrix}
$$

Toto produkuje **dokonale rovnomerné rozdelenie** cez SO(3) — každá možná rotácia je rovnako pravdepodobná. Toto je dôležité pre RL, pretože agent potrebuje vidieť všetky možné orientácie počas trénovania.

---

## Rýchla referencia

```python
# Naša konvencia kvaternionov: (w, x, y, z) — skalár prvý, ako MuJoCo

# Identita (žiadna rotácia)
q_identita = [1, 0, 0, 0]

# 90° okolo osi Z
q_90z = [cos(π/4), 0, 0, sin(π/4)] = [0.707, 0, 0, 0.707]

# Skladanie: najprv rotuj podľa q1, potom podľa q2
q_celkovy = quat_multiply(q2, q1)

# Inverzná rotácia
q_inv = quat_conjugate(q)  # = (w, -x, -y, -z)

# Chyba z aktuálnej do cieľovej
q_chyba = quat_multiply(q_ciel, quat_conjugate(q_aktualny))
chyba_os_uhol = axis_angle_from_quat(q_chyba)  # 3-D vektor
uhlova_vzdialenost = np.linalg.norm(chyba_os_uhol)  # skalár v [0, π]
```

---

**Ďalej:** [02 — Jacobián & Inverzná kinematika](02_jacobian_a_inverzna_kinematika.md) — ako používame tieto rotácie na riadenie robotického ramena.
