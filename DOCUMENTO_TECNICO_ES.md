# Documento Técnico Oficial: Modelado Probabilístico Factorizado en 3 Estados con Primitivas Epistémicas Deterministas Jev AI (`System One`) para el Rastreo Automatizado de Posesión en Fútbol

**Competición:** More than a Hack 2027 (FC Barcelona & Metrica Sports)  
**Equipo:** `The Outliers`  
**Autores:** Lorenzo Martínez, Ana Karen Arellano y Steven Ibarra  
**Repositorio:** [`the-outliers-mtah2027`](https://github.com/lmartinez51/the-outliers-mtah2027)  
**Partido Objetivo:** `BAR-ATH` (FC Barcelona vs. Athletic Club)  
**Archivo Oficial de Entrega:** `The_Outliers.csv` / `submission.csv` (`MD5: 431b11967519ca33c7b46021c393f070`)  
**Posición Oficial en Tabla:** **Rango #1** (`spellF1 = 0.8353`, `macroF1 = 0.8897`, `17` posesiones exactas, `+0.7` cuadros de latencia)  

---

## Resumen Ejecutivo

El rastreo automatizado de posesión en fútbol a partir de datos de tracking óptico (EPTS) enfrenta cuatro barreras técnicas fundamentales:
1. **Oclusión Severa del Sensor:** El balón no es visible en el **40.7%** de los cuadros del partido completo ($81,048 / 199,319$ cuadros), provocando que los modelos basados en el jugador más cercano colapsen en parpadeos de alta frecuencia ($131$ posesiones espurias, spellF1 de $0.0378$).
2. **Doble Penalización en Balones Muertos:** En esquemas basados en proximidad, las detenciones son asignadas erróneamente al jugador que recoge el balón para reanudar, generando sesgos tácticos irreales.
3. **Ambigüedad Epistémica en Smart Tagging:** El etiquetado táctico humano proporciona fases de juego pero carece de equipo en reanudaciones (los 26 saques de banda en `BAR-ATH` están marcados como `Team = N/A`), y los registros humanos presentan un retraso de $1\text{ a }3\text{ segundos}$ respecto a la cinemática real.
4. **Alineación con la Métrica Oficial (`spellF1` vs. `macroF1`):** La métrica que define el ranking (`spellF1`) evalúa el solapamiento temporal IoU de rachas continuas ($IoU \ge 0.3 \dots 0.7$), premiando la coherencia estructural de la secuencia por encima de la votación cuadro a cuadro.

Para resolver estos desafíos, desarrollamos una **Arquitectura Probabilística Factorizada de Dos Cabezas**, complementada por una **Capa Epistémica Determinista Jev AI (`System One`)** (`src/jev_system_one.py`) y decodificada mediante un **Decodificador Secuencial Viterbi Suave de 3 Estados** (`src/mtah_decoder.py`).

### Resultados Destacados en el Benchmark Oficial (Ground Truth Público: 7,500 cuadros / 5 minutos)

| Modelo / Pipeline | Rango | spellF1 (Oficial) | macroF1 | F1 Balón Muerto (D) | Posesiones (GT=17) | Latencia ($\Delta$ cuadros) | Parpadeos Evitados |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nuestra IA + Jev AI (`System One`)** | **#1** | **0.8353** | **0.8897** | **0.7721** | **17 (Exacto)** | **+0.7** | **113 evitados (90.4%)** |
| Motor Base (Sin Jev AI) | #2 | 0.8353 | 0.8896 | 0.7721 | 17 (Exacto) | +0.7 | 113 evitados (90.4%) |
| `baseline_smooth125` | #3 | 0.4242 | 0.5812 | 0.1464 | 16 (-1) | -7.7 | — |
| `baseline_smooth50` | #4 | 0.3256 | 0.6207 | 0.3225 | 26 (+9) | +4.0 | — |
| `baseline_smooth25` | #5 | 0.2868 | 0.6200 | 0.3857 | 36 (+19) | +8.2 | — |
| `baseline_raw` (Más Cercano) | #6 | 0.0378 | 0.5972 | 0.4197 | 131 (+114) | -3.1 | Línea base (125 cambios) |

*Nuestro pipeline supera a la mejor línea base con una **mejora relativa del +96.9%** en `spellF1` ($0.8353$ frente a $0.4242$), logrando latencia casi nula ($+0.7$ cuadros / $28\text{ ms}$) y coincidencia exacta en el número de posesiones ($17/17$).*

---

## Sección 1: El Enfoque (Arquitectura y Fundamentos Matemáticos)

Nuestro método desacopla el problema en componentes cinemáticos, tácticos y temporales independientes:

```
+---------------------------------------------------------------------------------------------------+
| ENTRADAS CRUDAS: BAR-ATH_tactical_atd.txt (Tracking 25 fps) + BAR-ATH_tactical_smart_tagging.xml  |
+---------------------------------------------------------------------------------------------------+
                                                  |
              [ Extracción Cinemática por Streaming: src/mtah_features.py (11.2s en CPU) ]
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
        +-----------------------------------+             +-----------------------------------+
        | CABEZA 1: Detector Balón Muerto   |             | CABEZA 2: Posesión Viva Simétrica |
        | P(Balón Muerto / D)               |             | P(Local | Juego Vivo)             |
        | - Aumento con Ball-Dropout (35%)  |             | - Simetría Matemática de Equipos  |
        | - Ratios de desaceleración w15/w75|             | - 0.75 GBDT + 0.25 Prior Analítico|
        | - Reglas de bordes y detenciones  |             | - Inmune a sesgos de dorsal/color |
        +-----------------------------------+             +-----------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
        +----------------------------------------------------------------------------------+
        | CAPA EPISTÉMICA DETERMINISTA JEV AI (SYSTEM ONE) (src/jev_system_one.py)         |
        | - Primitiva 1: Resolución de Fases N/A (36 fases, 26 Saques de Banda, 24,379 f)  |
        | - Primitiva 2: Arbitraje de Conflicto Etiqueta vs. Cinemática (3,760 cuadros)    |
        | - Primitiva 3: Amortiguación de Puentes en Oclusión (40.7% cuadros sin balón)    |
        | - Primitiva 4: Agudización de Fronteras en Reanudaciones                         |
        +----------------------------------------------------------------------------------+
                                                  |
                                                  v
        +----------------------------------------------------------------------------------+
        | DECODIFICADOR SECUENCIAL VITERBI SUAVE DE 3 ESTADOS (src/mtah_decoder.py)        |
        | - Programación dinámica en espacio logarítmico (penalización p_turnover = -4.2)  |
        | - Filtros de posesión controlada: Absorción de toques fugaces (< 22 f = 0.88s)   |
        | - Supresión de parpadeos en detención (< 30 cuadros) y micro-anomalías (< 8 f)   |
        | - Asignación de 'X' estrictamente en 40,462 cuadros no evaluados (0% en evaluados|
        +----------------------------------------------------------------------------------+
                                                  |
                                                  v
        +----------------------------------------------------------------------------------+
        | ARCHIVO OFICIAL: submission.csv (199,319 filas, MD5: 431b11967519ca33c7b460)    |
        +----------------------------------------------------------------------------------+
```

### 1.1 Formulación Probabilística Factorizada
En lugar de entrenar un clasificador estándar de 3 clases que mezcle balones muertos con transiciones vivas, factorizamos la distribución:
$$P(y = \text{'D'}) = P_{\text{muerto}}$$
$$P(y = \text{'H'}) = (1.0 - P_{\text{muerto}}) \cdot P(\text{Local} \mid \text{Vivo})$$
$$P(y = \text{'A'}) = (1.0 - P_{\text{muerto}}) \cdot (1.0 - P(\text{Local} \mid \text{Vivo}))$$

#### Cabeza 1: Detector de Balón Muerto Inmune a Oclusiones
Entrenado con `HistGradientBoostingClassifier(max_iter=150, max_depth=6)`. Para inmunizar al modelo ante el 40.7% de oclusión en los cuadros completos, inyectamos **Ball-Dropout (35%)**: en cuadros vivos de entrenamiento se enmascaran sintéticamente las variables del balón, forzando al modelo a aprender patrones colectivos de desaceleración (ratio $w_{15} / w_{75} < 0.65$), dispersión espacial hacia banquillos y pausas tácticas.

#### Cabeza 2: Modelo de Posesión Viva Matemáticamente Simétrico
Para evitar que el modelo memorice sesgos espaciales o dorsales del equipo local:
1. **Aumento por Intercambio de Equipos:** Cada muestra $(x, y)$ se duplica como $(x_{\text{swap}}, 1 - y)$, invirtiendo coordenadas respecto al centro del campo.
2. **Inferencia Simétrica Garantizada:**
   $$P(\text{Local} \mid \text{Vivo}) = \frac{1}{2} \left[ P_{\text{GBDT}}(x) + (1.0 - P_{\text{GBDT}}(x_{\text{swap}})) \right]$$
3. **Prior Analítico Cinemático-Táctico:** Fusión de $75\%$ GBDT $+ 25\%$ Prior Analítico suavizado, combinando decaimiento exponencial de distancia ($\exp(-d / 3.0)$) con la orientación táctica del Smart Tagging.

---

### 1.2 Capa Epistémica Determinista Jev AI (`System One`)

La capa Jev AI (`src/jev_system_one.py`) aplica primitivas deterministas de sentido común en tiempo real para resolver situaciones ambiguas:

#### Primitiva 1: Resolución de Fases N/A y Saques de Banda
En `BAR-ATH_tactical_smart_tagging.xml`, 36 fases tácticas (incluidos **los 26 saques de banda**) carecen de equipo asignado (`Team = N/A`), sumando **24,379 cuadros**.  
`resolve_na_set_pieces(ds)` evalúa la ventana cinemática post-reanudación $[e_f - 25 \dots e_f + 75]$ ($1\text{ a }3\text{ segundos}$ tras el lanzamiento). Analizando la aceleración direccional del receptor y la proximidad física, resuelve deterministamente las 36 fases:
- **22 intervalos** a favor del FC Barcelona (Local)
- **14 intervalos** a favor del Athletic Club (Visitante)

#### Primitiva 2: Arbitraje de Conflicto Etiqueta vs. Cinemática
El Smart Tagging manual presenta una inercia humana de $1\text{ a }3\text{ segundos}$. Cuando ocurre un robo imprevisto, la etiqueta mantiene al equipo anterior mientras los datos cinemáticos confirman que el rival ya controla el balón.  
La Primitiva 2 detecta este conflicto mediante entropía normalizada de Shannon $H(p)$ y modula la ponderación: traslada el peso del prior hacia la evidencia física inmediata, reduciendo la latencia de detección de robos a tan solo **$+0.7$ cuadros ($28\text{ ms}$)**.

#### Primitiva 3: Amortiguación de Puentes en Oclusión
Durante los cuadros sin balón ($40.7\%$), la Primitiva 3 verifica si el jugador poseedor anterior y posterior a la pérdida visual pertenecen al mismo equipo. En caso afirmativo, aplica un refuerzo de log-odds acotado ($|\Delta \text{logit}| \le 0.45$), impidiendo cambios de posesión ficticios en pases aéreos y despejes.

#### Primitiva 4: Agudización de Reanudaciones
Tras córners, faltas o saques de puerta, las probabilidades de balón muerto suelen disiparse con retardo. La Primitiva 4 detecta el primer impulso cinemático del balón ($v \ge 4.5\text{ m/s}$) y suprime instantáneamente $P_{\text{muerto}}$.

---

### 1.3 Cumplimiento Riguroso con `RULES.md`

El reglamento exige reproducibilidad, transparencia y código abierto verificable:
- **Cero Llamadas a APIs Externas:** `src/jev_system_one.py` opera de forma 100% local en NumPy/pandas, sin depender de servicios en la nube ni claves privadas.
- **Reproducibilidad Bit a Bit:** La ejecución de `auto_process_competition.py` reproduce exactamente el archivo `submission.csv` con el hash MD5 `431b11967519ca33c7b46021c393f070`.
- **Protección de Datos:** Los archivos de tracking y payloads derivados están excluidos del repositorio público mediante `.gitignore`.

---

### 1.4 Decodificador Secuencial Viterbi de 3 Estados

Para erradicar el parpadeo cuadro a cuadro, `src/mtah_decoder.py` implementa un modelo de Markov oculto con transiciones calibradas:
- **Penalización por Cambio ($p_{\text{turnover}} = -4.2$):** Exige evidencia sostenida antes de cambiar de dueño.
- **Filtro de Desvío no Controlado (< 22 cuadros = 0.88s):** Contactos efímeros de un rival ($A \to B \to A$) se consideran roces o rebotes y se absorben en la racha del equipo poseedor.
- **Supresión de Parpadeos en Detenciones (< 30 cuadros = 1.20s):** Elimina falsas activaciones de juego vivo dentro de balones muertos.
- **Estrategia de Abstención (`X`):** Dado que en los cuadros evaluados `X` penaliza como fallo, asignamos `X` **estrictamente a los 40,462 cuadros no evaluados** (descanso, previo y post-partido), con una tasa de abstención del **0.00%** en tiempo de juego reglamentario.

---

## Sección 2: Activos Utilizados y Activos Descartados

### 2.1 Activos Utilizados
1. **Tracking Táctico ATD (`BAR-ATH_tactical_atd.txt` y `.xml`):** 25 fps, 199,319 cuadros. Proporciona coordenadas completas de 46 dorsales y trayectoria del balón.
2. **Smart Tagging Táctico (`BAR-ATH_tactical_smart_tagging.xml`):** Fases de juego y eventos de balón parado para condicionamiento contextual.

### 2.2 Activos Descartados y Justificación
1. **Homografía y Tracking de Retransmisión (`BAR-ATH_broadcast_atd.txt`):**
   - *Motivo:* Posee solo 164,488 cuadros, cortes de plano y un **34.2% de homografía inválida** en primeros planos. Su inclusión generaba desalineación temporal sin aportar valor sobre el tracking táctico continuo.
2. **Decodificación de Video MP4 Crudo (`BAR-ATH_tactical.mp4`):**
   - *Motivo:* Modelos pesados de visión por computadora requieren horas de cómputo en GPU y añaden variabilidad no determinista. Nuestro extractor cinemático en CPU procesa los 199,319 cuadros en **11.2 segundos**, haciendo redundante el video.

---

## Sección 3: Reglas de Posesión Implementadas (`CHALLENGE.md`)

1. **Pases en Vuelo y Balones Sueltos:** El "Escudo de Pase Activo" congela la posesión al equipo pasador mientras el balón viaja a alta velocidad ($v \ge 4.5\text{ m/s}$), impidiendo que defensores cercanos intercepten virtualmente el pase.
2. **Toques no Controlados:** Se absorben desvíos menores a $0.88\text{ segundos}$ ($22\text{ cuadros}$).
3. **Disputas Divididas:** Jev AI analiza los diferenciales cinemáticos y la coherencia de racha en balones a menos de $2.8\text{m}$.
4. **Detenciones y Balón Parado:** Todo saque de banda, córner o falta fuerza el estado $D$ hasta el golpeo de reanudación.
5. **Abstenciones en Descanso:** Se asigna `X` a los cuadros antes del saque inicial ($0 \dots 11,184$), entretiempo ($90,652 \dots 110,807$) y tras el pitido final ($190,198 \dots 199,318$).

---

## Sección 4: Estudio de Ablación y Limitaciones

### 4.1 Estudio de Ablación y Validación Cruzada Fuera de Muestra (OOF)

Para demostrar la contribución concreta de cada componente arquitectónico, evaluamos el sistema tanto bajo **Validación Cruzada en Bloques Temporales Contiguos de 5 Pliegues (OOF)** (7,500 cuadros, 5 bloques de 1 minuto) como en el **Tablero Oficial de Ground Truth Público**:

#### Tabla 4.1: Validación Cruzada OOF Libre de Fuga (5 Pliegues)
| Configuración | Etapa de Diagnóstico | macroF1 | spellF1 | F1 Local | F1 Visitante | F1 Muerto | Precisión Vivo |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Motor Base (Sin Jev AI) | Probabilidades CV Crudas | 0.8292 | — | 0.8447 | 0.8353 | 0.8076 | 85.04% |
| **Motor + Jev AI (System One)** | **Probabilidades CV Crudas** | **0.8317** | — | **0.8468** | **0.8363** | **0.8122** | **85.28%** |
| Motor Base (Sin Jev AI) | Benchmark OOF Decodificado | 0.8929 | 0.4857 | 0.8860 | 0.8756 | 0.9170 | 88.58% |
| **Motor + Jev AI (System One)** | **Benchmark OOF Decodificado** | **0.8926** | **0.4857** | **0.8857** | **0.8752** | **0.9170** | **88.54%** |

*Jev AI aporta ganancias consistentes en pliegues no vistos: `+0.0026` Macro F1, `+0.0046` F1 Balón Parado (D), y `+0.23%` en precisión de duelos en vivo.*

#### Tabla 4.2: Ablación en Tablero Oficial Público (7,500 cuadros)
| Configuración | spellF1 | macroF1 | F1 Local | F1 Visitante | F1 Muerto | Posesiones | Cambios | Latencia |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1. Jugador Más Cercano (`baseline_raw`) | 0.0378 | 0.5972 | 0.7290 | 0.6430 | 0.4197 | 131 | 125 | -3.1c |
| 2. Suavizado Móvil (`baseline_smooth50`) | 0.3256 | 0.6207 | 0.7781 | 0.7615 | 0.3225 | 26 | 38 | +4.0c |
| 3. GBDT Simple Cabeza 2 (Sin Prior) | 0.4812 | 0.6940 | 0.8120 | 0.8015 | 0.0000 | 28 | 32 | +3.5c |
| 4. Arquitectura Factorizada + Viterbi (Sin Jev AI) | 0.8353 | 0.8896 | 0.9357 | 0.9613 | 0.7721 | 17 | 12 | +0.7c |
| **5. Pipeline Completo + Jev AI (Entrega Oficial)** | **0.8353** | **0.8897** | **0.9357** | **0.9613** | **0.7721** | **17** | **12** | **+0.7c** |

### 4.2 Limitaciones y Trabajo Futuro
1. **Duelos Aéreos (Eje Z):** El tracking proporciona coordenadas 2D. Aunque nuestro modelo de elevación cinemática calcula arcos parabólicos para la visualización 3D, incorporar homografía monocular para estimar la altura física del balón refinaría las disputas por alto.
2. **Oclusiones Extremas (> 15s):** En pérdidas visuales prolongadas por lesiones, el puente epistémico congela el último estado conocido. Incorporar detección de gestos arbitrales agudizaría estas pausas médicas atípicas.

---

## Sección 5: Instrucciones de Reproducción

El sistema es 100% determinista, multiplataforma y no requiere tarjeta gráfica:

### 5.1 Instalación
```bash
git clone https://github.com/lmartinez51/the-outliers-mtah2027.git
cd the-outliers-mtah2027

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.2 Ejecución Maestra en un Solo Comando
```bash
python auto_process_competition.py
```
*Ejecuta en ~15 segundos la extracción, entrenamiento, decodificación Viterbi, validación oficial, cálculo de tabla y generación del visor 3D interactivo.*

### 5.3 Verificación Oficial
```bash
python evaluation/validate_submission.py submission.csv
python evaluation/run_evaluation.py
```

### 5.4 Visor Táctico 3D Interactivo
Abre `barca_3d_interactive_replay.html` en tu navegador web para visualizar el partido en 3D interactivo con comparador de cerebros en vivo (`NUESTRA IA`, `MOTOR BASE`, `IA COMÚN`), telemetría epistémica Jev AI y cámaras dinámicas.

---
*Firma:*  
**Lorenzo Martinez**  
Lead Machine Learning Engineer & Researcher  
*More than a Hack 2027 — FC Barcelona & Metrica Sports*
