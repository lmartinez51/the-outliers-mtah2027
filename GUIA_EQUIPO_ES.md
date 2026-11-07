> **Aviso de estado (nov 2026).** Versiones anteriores de este repositorio ajustaron modelos y parámetros con la ventana pública de 5 minutos del ground truth. La aclaración de reglas de los organizadores (recibida en noviembre) no lo permite. Estamos rehaciendo la solución. Las métricas de documentos anteriores son en muestra y no deben leerse como resultados válidos; las afirmaciones de cumplimiento de RULES.md en esos documentos quedan retiradas hasta terminar el rediseño.

# Guía de Inicio Rápido y Onboarding para el Equipo (Español)

**Proyecto:** More than a Hack 2027 — FC Barcelona & Metrica Sports  
**Equipo:** `The Outliers`  
**Autores:** Lorenzo Martínez, Ana Karen Arellano y Steven Ibarra  
**Repositorio:** [`the-outliers-mtah2027`](https://github.com/lmartinez51/the-outliers-mtah2027)  
**Objetivo:** Permitir a cualquier compañero de equipo reproducir el pipeline completo, validar la entrega oficial y explorar el visor 3D interactivo en **menos de 3 minutos**.

---

## 1. Requisitos del Sistema

- **Sistema Operativo:** Windows, macOS o Linux.
- **Python:** Versión `3.9`, `3.10`, `3.11` o `3.12`.
- **Hardware:** CPU estándar (Intel Core i5 / AMD Ryzen 5 o superior).  
  *Nota importante:* **NO se necesita tarjeta gráfica (GPU)** ni instalación de CUDA. Todo el pipeline de ML opera con matrices vectorizadas y modelos ligeros en CPU.
- **Espacio en disco:** ~250 MB libres (el archivo comprimido de datos tácticos pesa ~35 MB).

---

## 2. Preparación de Datos (Solo Archivos Tácticos)

Descomprime únicamente los archivos de la carpeta táctica dentro del directorio `data/tactical/`:

```
the-outliers-mtah2027/
└── data/
    └── tactical/
        ├── BAR-ATH_tactical_atd.txt          <-- Tracking cinemático 25 fps (63 MB)
        ├── BAR-ATH_tactical_atd.xml          <-- Metadatos de jugadores y campo (4 KB)
        └── BAR-ATH_tactical_smart_tagging.xml <-- Fases tácticas y eventos (2.8 MB)
```

> [!IMPORTANT]
> **LOS VIDEOS `.mp4` DE 1.8 GB NO SON NECESARIOS.**  
> Nuestro extractor cinemático extrae las velocidades, aceleraciones y distancias directamente del tracking óptico en ~17 segundos, ahorrándote descargar gigabytes de video.

---

## 3. Instalación Rápida (Entorno Virtual)

Abre tu terminal en la raíz del repositorio y ejecuta:

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno virtual
# En Windows (PowerShell):
.\.venv\Scripts\activate
# En Linux o macOS:
source .venv/bin/activate

# 3. Instalar librerías requeridas
pip install -r requirements.txt
```

---

## 4. Ejecución del Pipeline en Un Solo Comando

Para ejecutar todo el pipeline de punta a punta (extracción de variables, entrenamiento de modelos, decodificación Viterbi, validación oficial, generación de tabla de posiciones y exportación del visor 3D):

```bash
python auto_process_competition.py
```

### ¿Qué hace este comando automáticamente?
1. **Paso 1 (Features):** Carga o procesa los 199,319 cuadros tácticos del partido.
2. **Paso 2 (Modelos + Jev AI):** Entrena el detector de balón muerto y el estimador simétrico de posesión viva con primitivas epistémicas deterministas.
3. **Paso 3 (Decodificación Viterbi):** Aplica programación dinámica para suprimir parpadeos falsos y genera `submission.csv` (`199,319` filas).
4. **Paso 4 (Validación y Leaderboard):** Ejecuta `evaluation/validate_submission.py` y `evaluation/run_evaluation.py`, confirmando el **Puesto #1** en la tabla oficial (`spellF1 = 0.9000`).
> ⚠️ *Cifras en muestra / fuera de muestra: no son un resultado válido. Ver el Aviso de estado al inicio de este documento.*

5. **Paso 5 (Visor 3D):** Compila el archivo autocontenido `barca_3d_interactive_replay.html` para inspección visual inmediata.

---

## 5. Opciones Adicionales del Master Orchestrator

| Comando / Bandera | Descripción |
|---|---|
| `python auto_process_competition.py --team-name "MiEquipo"` | Asigna el nombre de tu equipo, generando `<team_name>.csv` en la raíz y en `evaluation/results/` como exige `SUBMISSION.md`. |
| `python auto_process_competition.py --rebuild-cache` | Fuerza el reentrenamiento y reextracción completa desde los archivos crudos de `data/tactical/`. |
| `python auto_process_competition.py --no-jev` | Ejecuta la ablación del Motor Base (sin primitivas Jev AI) para comparar métricas. |
| `python auto_process_competition.py --3d-slice full --3d-stride 2` | Exporta el partido completo al visor 3D a 12.5 fps. |
| `python auto_process_competition.py --skip-3d` | Ejecuta únicamente el procesamiento numérico y la evaluación, omitiendo el render 3D. |

---

## 6. Visualización 3D: ¿Qué archivo usar? (`barca_3d_interactive_replay.html` vs. `src/viewer.html`)

El repositorio cuenta con dos archivos HTML para la visualización, cada uno diseñado para un propósito específico:

| Archivo | Rol en el Proyecto | ¿Cuándo usarlo? | Características |
|---|---|---|---|
| **`barca_3d_interactive_replay.html`** *(en la raíz)* | **Producto Final Autocontenido (Standalone Replay)** | **Para evaluar, presentar y compartir con el equipo o los jueces.** | **100% Offline y Portátil.** No necesita servidor local, ni conexión a Internet (librería Three.js y datos de 7,500 cuadros embebidos). Basta con **hacer doble clic**. |
| **`src/viewer.html`** *(en `src/`)* | **Plantilla de Código Fuente (Source Template)** | **Para desarrollo técnico e iteración en el visor.** | Carga los datos dinámicamente desde `src/match_payload.js`. Si modificas estilos CSS, luces 3D, cámaras o texturas, editas este archivo y luego ejecutas el generador para recompilar el standalone. |

> [!TIP]
> **Para el 99% de las revisiones del equipo y los jueces:** Abre directamente **`barca_3d_interactive_replay.html`** en la raíz del repositorio.

---

## 7. Simulación por Tiempos del Partido: Cómo Analizar Cualquier Segmento

Por defecto, el archivo `barca_3d_interactive_replay.html` incluye **7,500 cuadros (5 minutos, del cuadro 11,185 al 18,684)**, correspondientes exactamente a la **ventana de evaluación oficial con Ground Truth público**. Esto mantiene el archivo ultraligero (~5 MB), permite que abra de inmediato sin latencia y facilita la auditoría del concurso.

Si tus compañeros desean analizar **el primer tiempo, el segundo tiempo, los 90 minutos completos o cualquier ventana de minutos específica**, pueden exportar la simulación deseada con los siguientes ejemplos:

### 📌 Ejemplo 1: Primer Tiempo Completo (~45-53 min)
Exporta desde el saque inicial hasta el descanso (`cuadros 11,185 a 90,651`):
```bash
# Con el generador directo (recomendado stride 2 = 12.5 fps para máxima fluidez y peso óptimo):
python generate_3d_replay_html.py --half 1 --stride 2 --standalone

# O mediante el orquestador maestro:
python auto_process_competition.py --3d-slice half1 --3d-stride 2
```

### 📌 Ejemplo 2: Segundo Tiempo Completo (~45-53 min)
Exporta desde la reanudación del segundo tiempo hasta el silbatazo final (`cuadros 110,808 a 190,197`):
```bash
# Con el generador directo:
python generate_3d_replay_html.py --half 2 --stride 2 --standalone

# O mediante el orquestador maestro:
python auto_process_competition.py --3d-slice half2 --3d-stride 2
```

### 📌 Ejemplo 3: Todo el Partido Completo (90 minutos, 199,319 cuadros)
Exporta la totalidad de los datos tácticos registrados:
```bash
# Con el generador directo:
python generate_3d_replay_html.py --full --stride 2 --standalone

# O mediante el orquestador maestro:
python auto_process_competition.py --3d-slice full --3d-stride 2
```

### 📌 Ejemplo 4: Ventana de Minutos Personalizada (Ej. del minuto 20 al 30)
Para estudiar una jugada o tramo en particular, utiliza la fórmula:  
$$\text{cuadro} = \text{minuto} \times 60 \text{ s} \times 25 \text{ fps}$$
- **Minuto 20:** $20 \times 60 \times 25 = 30,000$
- **Minuto 30:** $30 \times 60 \times 25 = 45,000$

```bash
# Exportar segmento a 25 fps nativo con nombre de archivo personalizado:
python generate_3d_replay_html.py --start-frame 30000 --end-frame 45000 --stride 1 --standalone --output replay_min20_a_30.html
```

> [!TIP]
> **Recomendación de rendimiento (`--stride`):**
> - `--stride 1` (25 fps nativos): Máxima fidelidad cinemática, ideal para clips cortos de 5 a 10 minutos.
> - `--stride 2` (12.5 fps interpolados): Fluidez excelente reduciendo a la mitad el tamaño del archivo HTML y el consumo de memoria RAM del navegador. Recomendado para tiempos completos.

---

## 8. Manual del Visor Táctico 3D Interactivo

Para ver la repetición 3D, simplemente **haz doble clic** o abre `barca_3d_interactive_replay.html` en Google Chrome, Microsoft Edge, Mozilla Firefox o Safari.

### 🎮 Atajos de Teclado
- **`Barra Espaciadora`**: Pausar / Reanudar la animación.
- **`M`**: Alternar entre los 3 Cerebros de IA:
  1. `🧠 NUESTRA IA (+ Jev System One)`: Decodificación óptima con absorción de desvíos y resolución de saques de banda.
  2. `⚙️ MOTOR BASE (Sin Jev AI)`: Modelo base sin arbitraje epistémico de jugadas divididas.
  3. `🤖 IA COMÚN (Jugador Más Cercano)`: Simulación de lo que vería un aficionado con modelos ingenuos de proximidad (frecuentes errores de parpadeo).
- **`H`**: Activar / Desactivar **Modo Cine (Vista Limpia)**, ocultando los paneles laterales para observar el terreno completo.
- **`C`**: **Contraer / Expandir el panel inferior**, reduciéndolo verticalmente a solo la línea informativa principal para maximizar la visibilidad del terreno de juego.
- **`Flechas Izquierda / Derecha`**: Retroceder o avanzar 1 segundo.
- **`J` / `K`**: Saltar al momento clave anterior o siguiente (robos de balón y desempates Jev AI).
- **`1`**: Cámara de Transmisión TV (seguimiento dinámico de jugada).
- **`2`**: Cámara Dron Táctico (perspectiva elevada en diagonal).
- **`3`**: Cámara Aérea Zenital 2D (vista de pizarra táctica para entrenadores).
- **`4`**: Cámara Libre Orbital (arrastra con el ratón para rotar y usa la rueda para hacer zoom).

### 🎨 Elementos Visuales en el Terreno 3D
- **🔵🔴 Jugadores Azulgrana (Dorsal blanco/oro):** FC Barcelona atacando con sus nombres reales (Pedri, Lewandowski, Lamine Yamal, Cubarsí, Ter Stegen, etc.).
- **⚪🔴 Jugadores Albirrojos (Dorsal cian):** Athletic Club defendiendo / atacando con sus nombres reales (Nico Williams, Iñaki Williams, Sancet, Vivian, Padilla, etc.).
- **🟡 Baliza y Esferas de Energía:** Conectan al jugador poseedor con el balón mediante un hilo electromagnético tridimensional que nace a ras de césped desde las botas del jugador.
- **🛡️ Escudo de Pase 3D:** Un domo protector dorado que rodea el balón en pases rápidos ($v > 4.5\text{ m/s}$) para evidenciar cómo nuestra IA ignora defensores cercanos que no llegan a cortar la trayectoria.
- **🟣 Destello Violeta (Jev AI):** Se enciende cuando el árbitro epistémico desempata jugadas divididas cuerpo a cuerpo o reanudaciones de saque de banda sin asignación previa.
- **🛑 Tarjeta HUD Balón Muerto:** Cuando el juego se detiene por falta, córner o saque de banda, el panel lateral derecho se viste en tono ámbar neutral (`🛑 BALÓN MUERTO`), evitando atribuir posesiones ficticias al equipo que va a sacar.

---

## 9. Verificación de Entrega para la Competición

Antes de enviar el formulario oficial de la competición:
```bash
# Verificar integridad de formato
python evaluation/validate_submission.py submission.csv

# Verificar tabla de posiciones
python evaluation/run_evaluation.py
```

El resultado esperado en consola debe ser:
- **Filas:** `199,319`
- **Formato:** `OK submission.csv`
- **Spell F1:** `0.9000`
- **Macro F1:** `0.9772`
- **Posesiones:** `15` (de 17 en el Ground Truth público, 100% de precisión de rachas)
> ⚠️ *Cifras en muestra / fuera de muestra: no son un resultado válido. Ver el Aviso de estado al inicio de este documento.*

- **Abstenciones en juego evaluado:** `0 cuadros (0.00%)`

¡Cualquier duda o ajuste, comunícate con Lorenzo Martinez en el canal del equipo!
