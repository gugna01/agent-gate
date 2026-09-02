# agent-gate

**Una compuerta de validación para registros producidos por agentes de IA.**

El modo de falla que importa en un sistema con IA no es la caída. Es el agente
que devuelve algo **plausible y equivocado**: un nombre bien formado, una
categoría coherente, una cifra razonable. Ningún `try/except` detecta eso. Se
escribe en la base de datos, se ve normal, y contamina todo lo que venga
después.

Este repositorio implementa la respuesta en dos frentes: **una compuerta por la
que pasa todo registro, con tres salidas y ningún atajo** — y, sobre ella, un
sistema **RAG** que aplica la misma idea a las respuestas generadas.

```
extraer  ->  normalizar  ->  COMPUERTA  ->  aceptado
                                |            revisión humana
                                |            rechazado
                             fallo de extracción (ruidoso)
```

## Correlo ahora

Sin dependencias, sin clave de API, sin red.

```bash
python examples/run_demo.py
```

```
processed 6 document(s): 1 accepted, 1 queued for review, 2 rejected, 2 extraction failure(s)

ACCEPTED — written without a human in the loop
  Northwind Logistics  <ops@northwind-logistics.example>  mid_market  n=320
    origin: doc-001 via heuristic-v1 at 2026-08-31T19:03:55+00:00 [925ee7a7a140f450]

REVIEW — held back for a person
  - Beacon Analytics  (confidence 0.80 in review band [0.5, 0.85))

REJECTED — never reaches the store
  [rejected] Halyard Freight — schema: contact_email — malformed

EXTRACTION FAILURES — the agent said so instead of guessing
  doc-004: no company name found
```

Seis documentos entran, uno se escribe. **Esa proporción es el punto.** Un
pipeline que aceptara los seis se vería más productivo y estaría envenenando
la base de datos en silencio.

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q      # 61 tests
```

## Las cuatro decisiones

### 1. La estructura se valida antes que la confianza, nunca al revés

Un registro con 100 % de confianza y un correo malformado sigue siendo basura.
Una compuerta que mira el puntaje primero lo deja pasar con entusiasmo.

```python
def test_malformed_record_is_rejected_even_at_full_confidence(self):
    decision = Gate().evaluate(make(contact_email="nope", confidence=1.0))
    assert decision.verdict is Verdict.REJECTED
```

La validación es estructural y aburrida a propósito —formatos, rangos, campos
obligatorios— y **no le pregunta al modelo que produjo el candidato**. Un
modelo no puede ser juez de su propia salida.

### 2. La banda de revisión no es indecisión, es diseño

Entre el umbral de aceptación (0.85) y el piso de revisión (0.50) hay una
franja donde nada está mal y nada está confirmado. Esa franja es la parte del
problema en la que una persona es mejor que el sistema.

Un pipeline sin cola de revisión no elimina esos casos: **los adivina.**

La cola está diseñada desde el principio, no añadida después del primer lote
malo. Ese orden es la lección que este repositorio existe para demostrar.

### 3. Un campo ausente baja la confianza; no es una violación de esquema

Confundir "dato faltante" con "dato inválido" manda buenos registros a la
basura y hace que la compuerta parezca más estricta de lo que es. Un campo
opcional ausente es **evidencia faltante**, y eso pertenece al puntaje.

```python
def _to_int(value: object) -> int | None:
    """Return None rather than 0 on failure. Zero is a claim; None is silence."""
```

Este error estaba en la primera versión y lo encontró el propio demo: la cola
de revisión salía vacía porque un campo ausente se rechazaba por esquema.

### 4. Todo registro carga su origen, o no se escribe

```python
class ProvenanceMissing(AgentGateError):
    """A record reached the gate without a traceable origin."""
```

Cuando aparece un registro malo tres semanas después, poder auditar hacia
atrás es la única forma de aprender algo de él. Sin trazabilidad solo queda
borrarlo y esperar.

## Extracción determinista y extracción con modelo

Dos implementaciones, **la misma interfaz**:

- `HeuristicExtractor` — reglas, sin red. Su confianza es honesta por
  construcción: es la fracción de campos que realmente encontró. Nunca reporta
  una certeza que no tiene.
- `ModelExtractor` — recibe cualquier cliente de LLM como un simple callable.
  Impone tres cosas sobre el modelo: la salida debe parsear como JSON (un
  modelo que divaga es un fallo, no un éxito parcial), un nombre ausente es un
  fallo y no un campo que inventar, y **la confianza autoreportada se acota y
  sigue pasando por la misma compuerta**. La autoevaluación es una entrada,
  jamás un veredicto.

El pipeline no distingue cuál está conectado. Ese es el punto: **la maquinaria
de seguridad no depende del extractor.**

## Un bug que encontraron los tests

El prompt de `ModelExtractor` contiene JSON literal como ejemplo. Usar
`str.format()` sobre él hace que Python lea `{"error"}` como un marcador de
posición y lance `KeyError`.

```python
# str.replace, not str.format: the prompt contains literal JSON
# braces, and format() reads those as placeholders. Caught by a test,
# which is the only reason it is not a production incident.
raw = self._complete(self._PROMPT.replace("<<TEXT>>", text))
```

Lo dejo documentado porque es exactamente la clase de fallo que solo aparece
en la rama menos transitada —la de manejo de errores— y que en producción se
descubre el día que más duele.

## Estructura

```
src/agentgate/
  errors.py       tipos de fallo explícitos
  provenance.py   origen inmutable y huella del texto fuente
  schema.py       el registro objetivo y qué lo hace válido
  extractors.py   heurístico y basado en modelo, misma interfaz
  normalize.py    canonicalización, antes de la compuerta
  gate.py         la compuerta: tres salidas, sin atajos
  queue.py        cola de revisión humana
  pipeline.py     orquestación, deliberadamente delgada
examples/         demo ejecutable
tests/            61 tests
```

El pipeline no escribe en ningún almacén. Devuelve un informe y quien lo llama
decide. Eso lo hace probable sin base de datos y deja los efectos secundarios
a la vista.

---

# RAG con verificación de respaldo

`agentgate.rag` extiende la misma idea a preguntas y respuestas. La
recuperación decide **qué lee el modelo**; la compuerta decide **si lo que dijo
se puede rastrear hasta ahí**. Hacen falta las dos: un modelo con el contexto
correcto sigue pudiendo agregar una frase que no aparece en él.

```bash
python examples/run_rag_demo.py
```

```
P: ¿Qué cubre la garantía y cómo la activo?
[   review] groundedness 0.63 — 1 of 3 sentence(s) unsupported — the rest stands
  sources: garantia#0
  R: La garantía cubre defectos de fabricación durante doce meses desde la fecha de
     compra. Para activarla se requiere la factura de compra. Además, la garantía
     puede extenderse veinticuatro meses adicionales pagando una prima equivalente
     al quince por ciento del valor del producto. [garantia#0]
  ⚠ sin respaldo en el corpus: Además, la garantía puede extenderse veinticuatro
     meses adicionales pagando una prima equivalente al quince por ciento…
```

Dos frases correctas y una tercera **fluida, específica y falsa**. Está bien
escrita, suena a política de empresa y no aparece en ningún documento. Eso es
lo que hay que detectar, y se detecta frase por frase.

## Las decisiones

### Dónde se corta importa más que el tamaño

Un fragmento que termina a mitad de frase, o que separa una tabla de su
encabezado, recupera mal por bueno que sea el modelo de embeddings: el
significado se destruyó antes de que el modelo lo viera. El divisor respeta
párrafos primero, frases después, y prefiere pasarse del tamaño objetivo antes
que partir una frase.

El solapamiento de una frase entre fragmentos cuesta almacenamiento y compra
recuperación: un dato que cae justo en el borde no sería recuperable desde
ninguno de los dos lados.

### La morfología del español rompe el emparejamiento por palabras

Esto no fue una decisión de diseño: fue un fallo que encontró el demo.

La primera versión indexaba palabras. La pregunta *"¿Cuántos días tengo para
devolver un producto?"* recuperaba la **política de envíos** — porque ambas
mencionan "días" y ninguna comparte "devolver" con "devoluciones".

`devolver`, `devolución` y `devoluciones` son tres tokens distintos que no se
solapan en nada. La solución estándar son los n-gramas de caracteres: les da un
núcleo común (`devol`, `evolu`) y el emparejamiento sobrevive a la flexión.
Pesan menos que las palabras completas, porque a peso igual dos palabras largas
sin relación que comparten un fragmento empiezan a ganarle a una coincidencia
exacta.

```python
def test_char_ngrams_bridge_spanish_morphology(self):
    """The bug the demo exposed: 'devolver' must reach 'devoluciones'."""
    shared = set(char_ngrams("devolver")) & set(char_ngrams("devoluciones"))
    assert shared
```

### Si no hay con qué responder, no se llama al modelo

```python
if retrieval.verdict is RetrievalVerdict.INSUFFICIENT:
    return Answer(verdict=AnswerVerdict.REFUSED, ...)
```

Rechazar antes de llamar al modelo no solo es más seguro, es más barato. La
mayoría de los sistemas RAG descubre esto en su primera factura.

### Una cita que no se validó es decoración

Un modelo cita con total naturalidad un identificador que nunca vio. Las citas
se contrastan contra los fragmentos realmente recuperados; las inventadas se
descartan, y una respuesta sin cita válida no se entrega como respuesta: nadie
puede comprobarla.

### Rechazar solo cuando no queda nada rescatable

Enrutar por el promedio de la puntuación tira a la basura lo más útil que
produjo la verificación: **saber cuáles frases fallaron.** Una respuesta con
tres frases buenas y una inventada promedia hacia el rechazo, y se descarta una
respuesta correcta porque una cláusula era falsa — cuando un revisor
simplemente borraría esa cláusula.

Así que el rechazo se reserva para respuestas sin nada que salvar. Si alguna
frase se sostiene, decide una persona.

Este también salió de un test: esperaba revisión y recibió rechazo.

## Sobre la medida, sin exagerarla

La verificación de respaldo se calcula por **solapamiento léxico** entre cada
frase de la respuesta y los fragmentos recuperados, contando solo palabras con
contenido.

Detecta con fiabilidad el fallo común —el modelo inventa un nombre, una cifra o
una regla que no está en la fuente—. **No** detecta una paráfrasis fiel marcada
como no respaldada, ni una frase que reutiliza el vocabulario de la fuente
invirtiendo su sentido.

Un sistema en producción reemplaza esa función por un modelo NLI o un LLM como
juez, y no cambia nada a su alrededor. **La interfaz es lo que importa; la
medida es intercambiable.**

Lo mismo aplica a `HashingEmbedder`: es determinista y sin dependencias, que es
lo que hace este repositorio ejecutable por cualquiera sin clave de API. El
propio demo termina mostrando dónde se rompe:

```
LÍMITE CONOCIDO — recuperación léxica, misma pregunta en otras palabras

P: ¿En cuánto tiempo llega un pedido a una ciudad principal?
   politica-devoluciones#0  0.286
   envios#0                 0.273
```

La respuesta está en `envios#0`, pero la pregunta no comparte ni una palabra de
contenido con ella. Eso no se arregla ajustando umbrales: se cambia
`HashingEmbedder` por un modelo real de embeddings, que es para lo que existe
`ModelEmbedder`.

Prefiero mostrar el límite a que lo encuentre quien evalúe el código.

## Estructura

```
src/agentgate/rag/
  chunking.py     división que respeta párrafos y frases, con solapamiento
  embedding.py    hashing determinista con n-gramas de caracteres, y adaptador a modelo real
  index.py        índice vectorial y búsqueda por coseno
  retrieval.py    piso de puntuación: cuándo el corpus no alcanza
  answering.py    respuesta, verificación de respaldo y validación de citas
```

---

## Por qué existe este repositorio

Diseñé y llevé a producción un sistema multi-agente de búsqueda y clasificación
de datos como único ingeniero de una plataforma real. Las primeras versiones
escribían directo en la base de datos y generaban registros de proveedores
plausibles y equivocados. Lo que lo arregló fue exactamente el patrón que está
aquí: validación estructural antes de escribir, umbral de confianza explícito,
cola de revisión humana y fallo ruidoso en lugar de la mejor conjetura.

Este código es una reimplementación limpia de ese patrón en un dominio
inventado. **No contiene código, datos, esquemas ni prompts de ningún sistema
en producción.**

Cambió mi forma de trabajar con IA en general. La pregunta útil dejó de ser
"¿funciona?" y pasó a ser **"¿cómo sabría que esto está mal?"**.

## Licencia

Sin licencia de uso. Ver [NOTICE](NOTICE).

---

**Santiago Achuri Vargas** · [LinkedIn](https://www.linkedin.com/in/santiago-achuri-vargas-09723140b/) · santagochury@gmail.com
