# agent-gate

**Una compuerta de validación para registros producidos por agentes de IA.**

El modo de falla que importa en un sistema con IA no es la caída. Es el agente
que devuelve algo **plausible y equivocado**: un nombre bien formado, una
categoría coherente, una cifra razonable. Ningún `try/except` detecta eso. Se
escribe en la base de datos, se ve normal, y contamina todo lo que venga
después.

Este repositorio implementa la respuesta: **una compuerta por la que pasa todo
registro, con tres salidas y ningún atajo.**

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
python -m pytest tests -q      # 27 tests
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
tests/            27 tests
```

El pipeline no escribe en ningún almacén. Devuelve un informe y quien lo llama
decide. Eso lo hace probable sin base de datos y deja los efectos secundarios
a la vista.

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
