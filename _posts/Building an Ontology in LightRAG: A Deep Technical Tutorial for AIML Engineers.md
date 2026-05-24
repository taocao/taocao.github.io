# Building an Ontology in LightRAG: A Deep Technical Tutorial for AIML Engineers

## Why Ontology Matters: The Limits of Vector-Only RAG

Traditional Retrieval-Augmented Generation (RAG) works by embedding documents into a vector space and retrieving the top-k most similar chunks at query time. This works well for surface-level similarity — "find me text that looks like this question" — but it fundamentally fails at structural reasoning.

Consider a question like: *"What are all the organizations that Alice has worked with, and how do those organizations relate to each other?"* A vector search will find chunks mentioning Alice, but it cannot traverse the graph of relationships to answer the second part. It has no concept of *structure*.

This is the core problem LightRAG solves by building an **ontology** — a formal, machine-readable representation of entities and their relationships extracted from your documents. The ontology is a knowledge graph where nodes are entities (people, organizations, concepts, methods) and edges are typed, weighted relationships between them. This structure enables retrieval modes that go far beyond similarity search.

Understanding how this ontology is built is a foundational skill for any AIML engineer working with knowledge-intensive applications. This tutorial walks through every layer of the pipeline, from raw text to a queryable knowledge graph, with the actual production code from LightRAG.

---

## The Big Picture: Five Stages of Ontology Construction

Before diving into code, here is the end-to-end pipeline:

```
Raw Documents
     │
     ▼
[1] Document Chunking
     │  (split into overlapping token windows)
     ▼
[2] Prompt Configuration
     │  (entity type taxonomy + KG Specialist prompts)
     ▼
[3] LLM Extraction per Chunk
     │  (concurrent, semaphore-controlled)
     ▼
[4] Gleaning (Iterative Refinement)
     │  (additional LLM passes to catch missed items)
     ▼
[5] Three-Phase Merge Pipeline
     │  (deduplicate, summarize, upsert to graph + vector DB)
     ▼
Knowledge Graph + Vector Index
```

Each stage has specific engineering decisions that directly affect the quality and cost of your ontology. Let us walk through each one.

---

## Stage 1: Document Chunking — Setting the Extraction Window

Before any LLM call happens, documents are split into chunks. The chunk size is a critical hyperparameter: too large and the LLM misses entities buried in long context; too small and you lose relational context needed to extract meaningful relationships.

LightRAG defaults to 1200 tokens per chunk with 100 tokens of overlap:

```python
# lightrag/base.py (configuration defaults)
chunk_token_size = 1200
chunk_overlap_token_size = 100
``` [1](#4-0) 

The overlap is important: a relationship that spans a chunk boundary (entity A in the last sentence of chunk N, entity B in the first sentence of chunk N+1) will be captured because both chunks share those boundary tokens.

**AIML Engineer Takeaway**: For dense technical documents (research papers, legal contracts), consider reducing `chunk_token_size` to 800-1000 to keep each chunk focused. For narrative text, 1200-1500 is reasonable. The overlap should be at least 10% of chunk size to avoid boundary losses.

---

## Stage 2: Prompt Configuration — Teaching the LLM What to Extract

The quality of your ontology is directly determined by the quality of your extraction prompts. LightRAG uses a `PROMPTS` dictionary in `lightrag/prompt.py` as the central registry for all extraction templates.

### The Entity Type Taxonomy

The first critical component is the entity type taxonomy. This tells the LLM how to classify every entity it finds:

```python
# lightrag/prompt.py, line 18
PROMPTS["default_entity_types_guidance"] = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Person: Human individuals, real or fictional
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Organization: Companies, institutions, government bodies, groups
- Location: Geographic places (cities, countries, buildings, regions)
- Event: Occurrences, incidents, ceremonies, meetings
- Concept: Abstract ideas, theories, principles, beliefs
- Method: Procedures, techniques, algorithms, workflows
- Content: Creative or informational works (books, articles, films, reports)
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)"""
``` [2](#4-1) 

This taxonomy is not fixed. For a medical domain, you would add `Disease`, `Drug`, `Gene`, `ClinicalTrial`. For a legal domain, you would add `Statute`, `Jurisdiction`, `Precedent`. The taxonomy shapes the entire ontology structure.

### The Knowledge Graph Specialist System Prompt

The system prompt establishes the LLM's role and extraction rules. It is worth reading in full because every instruction has a specific engineering purpose:

```python
# lightrag/prompt.py, line 34
PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships...

---Instructions---
1. Entity Extraction:
   - entity_name: title case, consistent naming across entire extraction
   - entity_type: from the taxonomy, or Other
   - entity_description: concise, based solely on input text

2. Relationship Extraction:
   - Decompose multi-entity statements into binary relationships
   - source_entity, target_entity, relationship_keywords, relationship_description

3. Record Types:
   - entity rows: exactly 4 tuple parts
   - relation rows: exactly 5 tuple parts

4. Output Format:
   entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
   relation<|#|>source<|#|>target<|#|>keywords<|#|>description
...
"""
``` [3](#4-2) 

Key engineering decisions embedded in this prompt:

- **Consistent naming**: "Capitalize the first letter of each significant word (title case)" — this prevents `openai`, `OpenAI`, and `OPENAI` from becoming three separate nodes in the graph.
- **Binary decomposition**: "If a single statement describes a relationship involving more than two entities, decompose it into multiple binary relationships" — this keeps the graph structure clean and traversable.
- **Strict field counts**: entity rows must have exactly 4 fields, relation rows exactly 5 — this makes parsing deterministic and allows the parser to reject malformed output.
- **Undirected relationships**: "Treat all relationships as undirected unless explicitly stated otherwise" — prevents duplicate edges from `(A→B)` and `(B→A)` representing the same fact.

### Prompt Profile Resolution

The `resolve_entity_extraction_prompt_profile` function merges prompt components in a priority chain:

```python
# lightrag/prompt.py, line 847
def resolve_entity_extraction_prompt_profile(
    addon_params: Mapping[str, Any] | None,
    use_json: bool,
) -> EntityExtractionPromptProfile:
    default_profile = get_default_entity_extraction_prompt_profile()
    addon_params = addon_params or {}
    prompt_file = addon_params.get("entity_type_prompt_file")

    # Priority: runtime addon_params > YAML file > built-in defaults
    guidance = addon_params.get("entity_types_guidance")
    if guidance is None:
        guidance = file_profile.get(
            "entity_types_guidance", default_profile["entity_types_guidance"]
        )
``` [4](#4-3) 

**AIML Engineer Takeaway**: You can customize the ontology schema at three levels without touching core code:
1. Pass `addon_params={"entity_types_guidance": "- Drug: ..."}` at runtime
2. Create a YAML profile file and set `ENTITY_TYPE_PROMPT_FILE=entity_type_prompt.yml`
3. Override the entire `PROMPTS["entity_extraction_system_prompt"]` string

---

## Stage 3: LLM Extraction — The Core Extraction Loop

The extraction entry point is `extract_entities` in `lightrag/operate.py`. This function receives all document chunks and orchestrates concurrent LLM calls:

```python
# lightrag/operate.py, line 3232
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    ...
) -> list:
``` [5](#4-4) 

### Concurrency Control

The function creates a semaphore to limit parallel LLM calls, preventing API rate limit errors:

```python
# lightrag/operate.py, line 3589
semaphore = asyncio.Semaphore(llm_model_max_async)
```

Each chunk is processed by `_process_single_content`, which first strips internal multimodal markup (image references, equation tags, drawing annotations) that would confuse the LLM:

```python
# lightrag/operate.py, line 3334
content = strip_internal_multimodal_markup_for_extraction(chunk_dp["content"])
``` [6](#4-5) 

### The LLM Call

The actual extraction call uses a caching wrapper to avoid re-processing chunks that have already been extracted (critical for incremental document ingestion):

```python
# lightrag/operate.py, line 3364
final_result, timestamp = await use_llm_func_with_cache(
    entity_extraction_user_prompt,
    use_llm_func,
    system_prompt=entity_extraction_system_prompt,
    llm_response_cache=llm_response_cache,
    cache_type="extract",
    chunk_id=chunk_key,
    response_format=({"type": "json_object"} if use_json_extraction else None),
)
``` [7](#4-6) 

### Two Output Modes: Delimiter vs JSON

LightRAG supports two output formats, controlled by `ENTITY_EXTRACTION_USE_JSON`:

```python
# lightrag/operate.py, line 3381
if use_json_extraction:
    maybe_nodes, maybe_edges = await _process_json_extraction_result(
        final_result, chunk_key, timestamp, file_path,
    )
else:
    maybe_nodes, maybe_edges = await _process_extraction_result(
        final_result, chunk_key, timestamp, file_path,
        tuple_delimiter=context_base["tuple_delimiter"],
        completion_delimiter=context_base["completion_delimiter"],
    )
``` [8](#4-7) 

**Delimiter mode** uses `<|#|>` as a field separator and `<|COMPLETE|>` as a terminator. It is faster but more fragile — LLMs sometimes insert extra delimiters or omit them.

**JSON mode** requests a structured JSON object. It is more reliable with smaller models but incurs higher latency because JSON-constrained generation is slower.

**AIML Engineer Takeaway**: Use JSON mode (`ENTITY_EXTRACTION_USE_JSON=true`) when working with models below 70B parameters. For large frontier models (GPT-4, Claude 3.5, Llama 3.1 70B+), delimiter mode is faster and equally reliable.

### Parsing and Validation

The delimiter-mode parser calls `_handle_single_entity_extraction` for each record:

```python
# lightrag/operate.py, line 421
def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        # Log malformed output and skip
        return None

    entity_name = sanitize_and_normalize_extracted_text(
        record_attributes[1], remove_inner_quotes=True
    )

    # Validate: empty name after sanitization → skip
    if not entity_name or not entity_name.strip():
        return None

    entity_type = sanitize_and_normalize_extracted_text(
        record_attributes[2], remove_inner_quotes=True
    )

    # Validate: reject types with special characters
    if not entity_type.strip() or any(
        char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
    ):
        return None

    # Handle comma-separated types: take first non-empty token
    if "," in entity_type:
        tokens = [t.strip() for t in entity_type.split(",")]
        entity_type = [t for t in tokens if t][0]

    # Normalize: lowercase, no spaces
    entity_type = entity_type.replace(" ", "").lower()

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
        file_path=file_path,
        timestamp=timestamp,
    )
``` [9](#4-8) 

Notice the defensive programming: every field is sanitized, validated, and normalized. LLMs produce inconsistent output — this parser is the quality gate that prevents dirty data from entering the knowledge graph.

---

## Stage 4: Gleaning — Iterative Refinement for Higher Recall

A single LLM pass over a chunk will miss entities. This is a known limitation of LLM extraction — models tend to focus on the most salient entities and skip secondary ones. LightRAG addresses this with **gleaning**: additional LLM passes that ask "did you miss anything?"

```python
# lightrag/operate.py, line 3399
run_gleaning = entity_extract_max_gleaning > 0
if run_gleaning and extract_tokenizer is not None and max_extract_input_tokens > 0:
    gleaning_token_count = (
        len(extract_tokenizer.encode(entity_extraction_system_prompt))
        + sum(len(extract_tokenizer.encode(msg.get("content", "") or ""))
              for msg in history)
        + len(extract_tokenizer.encode(entity_continue_extraction_user_prompt))
    )
    if gleaning_token_count > max_extract_input_tokens:
        # Skip gleaning: would exceed context window
        run_gleaning = False
``` [10](#4-9) 

The gleaning pass sends the original extraction as conversation history and appends a "continue" instruction. The LLM sees what it already extracted and is asked to find anything it missed. The token guard is critical: the gleaning prompt includes the full history, which can be large. If the combined token count would exceed `MAX_EXTRACT_INPUT_TOKENS` (default: 20480), gleaning is skipped rather than causing a context overflow error.

**AIML Engineer Takeaway**: `entity_extract_max_gleaning=1` (the default) gives a good recall/cost tradeoff. Setting it to 0 reduces LLM calls by ~50% at the cost of lower recall. Setting it to 2+ gives diminishing returns and significantly increases cost. For production systems with cost constraints, start with 1 and measure recall on a held-out evaluation set.

---

## Stage 5: Entity Merging — Consolidating Across Chunks

After extraction, the same entity (e.g., "OpenAI") may appear in dozens of chunks, each with a slightly different description. The merge pipeline consolidates these into a single authoritative node.

### The Merge Function

```python
# lightrag/operate.py, line 1912
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    ...
):
    # Step 1: Fetch existing node from graph
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        already_entity_types.append(already_node.get("entity_type"))
        already_source_ids.extend(existing_source_id.split(GRAPH_FIELD_SEP))
        already_description.extend(existing_desc.split(GRAPH_FIELD_SEP))

    # Step 2: Merge source chunk IDs
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)
``` [11](#4-10) 

### Deduplication and Type Resolution

After collecting all descriptions from all chunks, the function deduplicates them and resolves the entity type by majority vote:

```python
# lightrag/operate.py, line 2054
unique_nodes = {}
for i, dp in enumerate(nodes_data, start=1):
    desc = dp.get("description")
    if not desc:
        continue
    if desc not in unique_nodes:
        unique_nodes[desc] = dp
``` [12](#4-11) 

### Map-Reduce Description Summarization

When an entity appears in many chunks, the combined descriptions can exceed the LLM's context window. LightRAG uses a map-reduce strategy:

```python
# lightrag/operate.py, line 2088
description, llm_was_used = await _handle_entity_relation_summary(
    "Entity",
    entity_name,
    description_list,
    GRAPH_FIELD_SEP,
    global_config,
    llm_response_cache,
)
``` [13](#4-12) 

The `_handle_entity_relation_summary` function implements a multi-level summarization strategy:

```python
# lightrag/operate.py, line 198
async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    separator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """
    1. If total tokens < summary_context_size and count < force_llm_summary_on_merge:
       → just join descriptions, no LLM call
    2. If total tokens < summary_max_tokens:
       → summarize with LLM directly
    3. Otherwise:
       → split into chunks, summarize each (Map)
       → recursively summarize the summaries (Reduce)
       → continue until within token limits
    """
``` [14](#4-13) 

This is a classic MapReduce pattern applied to text summarization. For an entity that appears in 500 chunks, you cannot fit all 500 descriptions into a single LLM call. Instead:
- **Map**: Split into groups of N descriptions, summarize each group
- **Reduce**: Recursively summarize the summaries until you have one final description

**AIML Engineer Takeaway**: The `summary_context_size` and `force_llm_summary_on_merge` parameters control when LLM summarization kicks in. For cost optimization, increase `force_llm_summary_on_merge` to avoid LLM calls for entities with few descriptions. For quality, decrease it to ensure even small entities get LLM-quality summaries.

### Dual Storage Write

After merging, the entity is written to both storage backends:

```python
# lightrag/operate.py, line 2204
await knowledge_graph_inst.upsert_node(
    entity_name,
    node_data=node_data,
)

# lightrag/operate.py, line 2225
await safe_vdb_operation_with_exception(
    operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
    operation_name="entity_upsert",
)
``` [15](#4-14) [16](#4-15) 

The **graph storage** (`BaseGraphStorage`) stores the structural node with its description, type, and source IDs. The **vector storage** (`BaseVectorStorage`) stores the embedding of the entity description for semantic search. This dual storage is what enables LightRAG's hybrid retrieval modes.

---

## Stage 6: Relationship Merging — Building the Graph Edges

Relationships follow the same merge pattern as entities, with one additional concern: **weight aggregation**.

```python
# lightrag/operate.py, line 2271
if await knowledge_graph_inst.has_edge(src_id, tgt_id):
    already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)

# lightrag/operate.py, line 2390
weight = sum([dp["weight"] for dp in edges_data] + already_weights)
``` [17](#4-16) [18](#4-17) 

The weight of a relationship is the sum of weights from all extractions. If "Alice works at Acme" is mentioned in 10 different chunks, the edge weight is 10. This weight is used during global retrieval to rank relationships by their prominence in the corpus.

### Ensuring Graph Integrity

A relationship references two endpoint nodes. If a relationship is extracted but one of its endpoint entities was not extracted (perhaps it was filtered out during entity validation), the pipeline creates a minimal placeholder node:

```python
# lightrag/operate.py, line 2548
for need_insert_id in [src_id, tgt_id]:
    existing_node = await knowledge_graph_inst.get_node(need_insert_id)
    if existing_node is None:
        await knowledge_graph_inst.upsert_node(
            need_insert_id, node_data=node_data
        )
``` [19](#4-18) 

This ensures the graph is always structurally valid — every edge has both endpoints as nodes.

---

## Stage 7: The Three-Phase Parallel Merge Pipeline

The `merge_nodes_and_edges` function coordinates the entire merge process:

```python
# lightrag/operate.py, line 2826
async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    ...
) -> None:
    """
    Phase 1: Process all entities concurrently
    Phase 2: Process all relationships concurrently (may add missing entities)
    Phase 3: Update full_entities and full_relations storage
    """
``` [20](#4-19) 

### Phase 1: Concurrent Entity Processing with Keyed Locks

```python
# lightrag/operate.py, line 2877
all_nodes = defaultdict(list)
all_edges = defaultdict(list)

for i, (maybe_nodes, maybe_edges) in enumerate(chunk_results, start=1):
    for entity_name, entities in maybe_nodes.items():
        all_nodes[entity_name].extend(entities)
    for edge_key, edges in maybe_edges.items():
        sorted_edge_key = tuple(sorted(edge_key))  # undirected: sort for consistency
        all_edges[sorted_edge_key].extend(edges)
``` [21](#4-20) 

Then entities are processed concurrently with keyed locks:

```python
# lightrag/operate.py, line 2911
async def _locked_process_entity_name(entity_name, entities):
    async with semaphore:
        async with get_storage_keyed_lock(
            [entity_name], namespace=namespace, enable_logging=False
        ):
            entity_data = await _merge_nodes_then_upsert(
                entity_name, entities, ...
            )
``` [22](#4-21) 

The **keyed lock** is the critical concurrency primitive here. A global lock would serialize all entity writes, destroying throughput. A keyed lock allows concurrent writes to *different* entities while serializing concurrent writes to the *same* entity. If two chunks both extracted "OpenAI", their merge operations are serialized by the keyed lock on `"OpenAI"`, while "Microsoft" and "Google" are processed in parallel.

### Phase 2: Concurrent Relationship Processing

```python
# lightrag/operate.py, line 3016
async def _locked_process_edges(edge_key, edges):
    async with semaphore:
        async with get_storage_keyed_lock(
            sorted_edge_key, namespace=namespace, enable_logging=False,
        ):
            edge_data = await _merge_edges_then_upsert(
                edge_key[0], edge_key[1], edges, ...
            )
``` [23](#4-22) 

Relationships are processed after all entities because relationship merging may create new entity nodes as a side effect (the endpoint guarantee). Processing entities first minimizes the number of placeholder nodes created.

### Phase 3: Document Index Update

```python
# lightrag/operate.py, line 3157
if full_entities_storage and full_relations_storage and doc_id:
    await full_entities_storage.upsert({
        doc_id: {
            "entity_names": list(final_entity_names),
            "count": len(final_entity_names),
        }
    })
``` [24](#4-23) 

This document-level index maps each document to its extracted entities and relationships. It enables efficient document deletion (remove all entities/relationships that only appear in this document) and rebuild operations.

### Error Handling: Fail Fast

```python
# lightrag/operate.py, line 2976
done, pending = await asyncio.wait(
    entity_tasks, return_when=asyncio.FIRST_EXCEPTION
)
``` [25](#4-24) 

`FIRST_EXCEPTION` semantics mean the pipeline fails immediately if any entity merge fails, rather than silently continuing with a partially-written graph. This is the correct behavior for a data pipeline — partial writes are worse than no writes because they leave the graph in an inconsistent state.

---

## Configuration Reference for AIML Engineers

Here is a consolidated reference of the parameters that most directly affect ontology quality and cost:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_token_size` | 1200 | Larger = more context per extraction, but higher LLM cost |
| `chunk_overlap_token_size` | 100 | Larger = fewer boundary misses, but more redundant extraction |
| `entity_extract_max_gleaning` | 1 | 0 = fast/cheap, 1 = balanced, 2+ = high recall/expensive |
| `ENTITY_EXTRACTION_USE_JSON` | false | true = more reliable with small models, higher latency |
| `MAX_EXTRACTION_RECORDS` | 100 | Cap on total entities+relations per LLM response |
| `MAX_EXTRACTION_ENTITIES` | 40 | Cap on entity rows per LLM response |
| `force_llm_summary_on_merge` | varies | Threshold for triggering LLM summarization during merge |
| `summary_context_size` | varies | Max tokens before map-reduce kicks in | [26](#4-25) 

---

## Domain Customization: Owning Your Ontology Schema

The most important skill for an AIML engineer working with LightRAG is customizing the ontology schema for their domain. Here is a complete example for a biomedical domain:

```yaml
# prompts/entity_type_prompt.yml
entity_types_guidance: |
  - Gene: DNA sequences encoding proteins or functional RNA
  - Protein: Amino acid chains with biological function
  - Disease: Pathological conditions affecting organisms
  - Drug: Chemical compounds used for therapeutic purposes
  - Pathway: Biological signaling or metabolic pathways
  - ClinicalTrial: Registered studies testing interventions
  - Organism: Biological species or strains
  - Variant: Genetic mutations or polymorphisms

entity_extraction_examples:
  - input: "BRCA1 mutations increase the risk of breast cancer..."
    output: |
      entity<|#|>BRCA1<|#|>gene<|#|>BRCA1 is a tumor suppressor gene...
      entity<|#|>Breast Cancer<|#|>disease<|#|>Breast cancer is a malignant...
      relation<|#|>BRCA1<|#|>Breast Cancer<|#|>increases risk<|#|>BRCA1 mutations...
      <|COMPLETE|>
```

Then load it at runtime:

```python
rag = LightRAG(
    working_dir="./biomedical_kg",
    addon_params={
        "entity_type_prompt_file": "entity_type_prompt.yml",
        # Or override inline:
        # "entity_types_guidance": "- Gene: ...\n- Protein: ..."
    }
)
``` [27](#4-26) 

The `resolve_entity_extraction_prompt_profile` function handles the priority merge: your YAML file overrides the defaults, and any `addon_params` at runtime override the YAML file. [28](#4-27) 

---

## Key Engineering Lessons

Working through this codebase reveals several patterns that apply broadly to production AIML systems:

**1. Defensive parsing over trust**: The LLM output parsers (`_handle_single_entity_extraction`, `_handle_single_relationship_extraction`) validate every field, reject malformed records, and log warnings rather than crashing. In production, LLMs produce malformed output regularly — your parser must be the quality gate.

**2. Idempotent upserts over inserts**: Every write to the graph uses `upsert_node` and `upsert_edge`, not insert. This means re-processing a document is safe — it will merge with existing data rather than creating duplicates. This is essential for incremental ingestion pipelines.

**3. Keyed locks over global locks**: The `get_storage_keyed_lock` pattern allows maximum concurrency while preventing race conditions on shared entities. This is a pattern worth adopting in any concurrent data pipeline.

**4. Caching at the LLM call boundary**: `use_llm_func_with_cache` caches extraction results by chunk content hash. Re-indexing a document that has not changed costs zero LLM calls. This is critical for cost management in production.

**5. Map-reduce for unbounded inputs**: The `_handle_entity_relation_summary` map-reduce pattern handles entities that appear in arbitrarily many chunks without hitting context limits. This pattern generalizes to any problem where you need to aggregate LLM outputs over a large number of inputs.

---

## Summary: The Ontology as a First-Class Artifact

The ontology LightRAG builds is not a side effect of document ingestion — it is the primary artifact. Every design decision in the pipeline (chunk size, entity taxonomy, gleaning passes, merge strategy, dual storage) is oriented toward producing a high-quality, consistent, queryable knowledge graph.

For AIML engineers, the key skills to own are:

1. **Schema design**: Define entity types that match your domain's conceptual structure
2. **Prompt engineering**: Customize the extraction prompts with domain-specific examples
3. **Parameter tuning**: Balance recall (gleaning, chunk overlap) against cost (LLM calls, token limits)
4. **Quality evaluation**: Build an evaluation set of documents with known entities/relationships to measure extraction recall and precision
5. **Incremental maintenance**: Understand the merge pipeline so you can safely add, update, and delete documents without corrupting the graph

The code in `lightrag/operate.py` and `lightrag/prompt.py` is the complete implementation of these concepts. Reading it alongside this tutorial gives you the full picture from design intent to production implementation. [29](#4-28) [30](#4-29)

### Citations

**File:** docs/ProgramingWithCore.md (L76-78)
```markdown
| **chunk_token_size** | `int` | Maximum token size per chunk when splitting documents | `1200` |
| **chunk_overlap_token_size** | `int` | Overlap token size between two chunks when splitting documents | `100` |
| **tokenizer** | `Tokenizer` | The function used to convert text into tokens (numbers) and back using .encode() and .decode() functions following `TokenizerInterface` protocol. If you don't specify one, it will use the default Tiktoken tokenizer. | `TiktokenTokenizer` |
```

**File:** lightrag/prompt.py (L9-32)
```python
PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Default entity type guidance injected into extraction prompts via {entity_types_guidance}.
# Users can override this by passing entity_types_guidance in addon_params, or by
# replacing the full prompt template string in PROMPTS.
PROMPTS[
    "default_entity_types_guidance"
] = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Person: Human individuals, real or fictional
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Organization: Companies, institutions, government bodies, groups
- Location: Geographic places (cities, countries, buildings, regions)
- Event: Occurrences, incidents, ceremonies, meetings
- Concept: Abstract ideas, theories, principles, beliefs
- Method: Procedures, techniques, algorithms, workflows
- Content: Creative or informational works (books, articles, films, reports)
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)"""
```

**File:** lightrag/prompt.py (L34-95)
```python
PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` section of user prompt.

---Instructions---
1. **Entity Extraction:**
  - Identify clearly defined and meaningful entities in the `---Input Text---` section of user prompt.
  - For each entity, extract:
    - `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
    - `entity_type`: Categorize the entity using the type guidance provided in the `---Entity Types---` section below. If none of the provided entity types apply, classify it as `Other`.
    - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.

2. **Relationship Extraction:**
  - Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
  - If a single statement describes a relationship involving more than two entities, decompose it into multiple binary relationships.
  - For each binary relationship, extract:
    - `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `relationship_keywords`: One or more high-level keywords summarizing the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
    - `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.

3. **Record Types:**
  - `entity` is used only for entity rows and those rows always contain exactly 4 tuple parts total.
  - `relation` is used only for relationship rows and those rows always contain exactly 5 tuple parts total.
  - A row with two entity names plus relationship keywords and a relationship description must start with `relation`, never `entity`.
  - After the last entity row, switch prefixes to `relation` for every relationship row.

4. **Output Format:**
  - Entity row: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
  - Relation row: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`
  - Wrong: `entity{tuple_delimiter}Alice{tuple_delimiter}Acme{tuple_delimiter}founded{tuple_delimiter}Alice founded Acme`
  - Correct: `relation{tuple_delimiter}Alice{tuple_delimiter}Acme{tuple_delimiter}founded{tuple_delimiter}Alice founded Acme`

5. **Delimiter Usage:**
  - The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
  - Incorrect: `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
  - Correct: `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

6. **Output Order & Deduplication:**
  - Output all extracted entities first, followed by all extracted relationships.
  - Output at most {max_total_records} total rows across entities and relationships in this response.
  - Output at most {max_entity_records} entity rows in this response.
  - Output fewer rows if fewer high-value items are present. Do not try to fill the limit.
  - Only output relationship rows whose source and target entities are both included in the selected entity rows for this response.
  - If the limit is reached, stop adding new rows immediately and output `{completion_delimiter}`.
  - Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
  - Avoid outputting duplicate relationships.
  - Within the list of relationships, output the relationships that are **most significant** to the core meaning of the input text first.

7. **Context & Language:**
  - Ensure all entity names and descriptions are written in the **third person**.
  - Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.
  - The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8. **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships have been completely extracted and outputted.

---Entity Types---
{entity_types_guidance}

---Examples---
{examples}
"""
```

**File:** lightrag/prompt.py (L847-898)
```python
def resolve_entity_extraction_prompt_profile(
    addon_params: Mapping[str, Any] | None,
    use_json: bool,
) -> EntityExtractionPromptProfile:
    """Resolve and merge the configured entity extraction prompt profile."""

    default_profile = get_default_entity_extraction_prompt_profile()
    addon_params = addon_params or {}
    prompt_file = addon_params.get("entity_type_prompt_file")

    file_profile: dict[str, Any] = {}
    if prompt_file:
        prompt_path = resolve_entity_type_prompt_path(prompt_file)
        file_profile = load_entity_extraction_prompt_profile(prompt_path)
        required_examples_key = (
            "entity_extraction_json_examples"
            if use_json
            else "entity_extraction_examples"
        )
        if required_examples_key not in file_profile:
            mode_name = "json" if use_json else "text"
            raise ValueError(
                f"ENTITY_TYPE_PROMPT_FILE '{prompt_file}' must define "
                f"'{required_examples_key}' when entity extraction runs in "
                f"{mode_name} mode."
            )

    guidance = addon_params.get("entity_types_guidance")
    if guidance is None:
        guidance = file_profile.get(
            "entity_types_guidance", default_profile["entity_types_guidance"]
        )
    elif not isinstance(guidance, str) or not guidance.strip():
        raise ValueError(
            "addon_params['entity_types_guidance'] must be a non-empty string."
        )

    return {
        "entity_types_guidance": guidance,
        "entity_extraction_examples": list(
            file_profile.get(
                "entity_extraction_examples",
                default_profile["entity_extraction_examples"],
            )
        ),
        "entity_extraction_json_examples": list(
            file_profile.get(
                "entity_extraction_json_examples",
                default_profile["entity_extraction_json_examples"],
            )
        ),
    }
```

**File:** lightrag/operate.py (L198-213)
```python
async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    separator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge
```

**File:** lightrag/operate.py (L421-500)
```python
def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 fields on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.info(
                f"Empty entity name found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Handle comma-separated entity types by finding the first non-empty token
        if "," in entity_type:
            original = entity_type
            tokens = [t.strip() for t in entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            if not non_empty:
                logger.warning(
                    f"Entity extraction error: all tokens empty after comma-split: '{original}'"
                )
                return None
            entity_type = non_empty[0]
            logger.warning(
                f"Entity type contains comma, taking first non-empty token: '{original}' -> '{entity_type}'"
            )

        # Remove spaces and convert to lowercase
        entity_type = entity_type.replace(" ", "").lower()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
```

**File:** lightrag/operate.py (L1912-1981)
```python
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    timing_start = time.perf_counter()
    try:
        already_entity_types = []
        already_source_ids = []
        already_description = []
        already_file_paths = []

        # 1. Get existing node data from knowledge graph
        already_node = await knowledge_graph_inst.get_node(entity_name)
        if already_node:
            existing_entity_type = already_node.get("entity_type")
            # Coerce to str before any string operations: non-string values from
            # API/custom graph paths would otherwise raise TypeError on the comma check.
            if (
                not isinstance(existing_entity_type, str)
                or not existing_entity_type.strip()
            ):
                existing_entity_type = "UNKNOWN"
            # Sanitize entity_type read back from DB to prevent dirty data from propagating
            if "," in existing_entity_type:
                original = existing_entity_type
                tokens = [t.strip() for t in existing_entity_type.split(",")]
                non_empty = [t for t in tokens if t]
                existing_entity_type = non_empty[0] if non_empty else "UNKNOWN"
                logger.warning(
                    f"Entity type read from DB contains comma, taking first non-empty token: '{original}' -> '{existing_entity_type}'"
                )
            already_entity_types.append(existing_entity_type)

            existing_source_id = already_node.get("source_id") or ""
            already_source_ids.extend(existing_source_id.split(GRAPH_FIELD_SEP))

            existing_file_path = already_node.get("file_path") or "unknown_source"
            already_file_paths.extend(existing_file_path.split(GRAPH_FIELD_SEP))

            existing_desc = (already_node.get("description") or "").strip()
            if existing_desc:
                already_description.extend(existing_desc.split(GRAPH_FIELD_SEP))

        new_source_ids = [dp["source_id"] for dp in nodes_data if dp.get("source_id")]

        existing_full_source_ids = []
        if entity_chunks_storage is not None:
            stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
            if stored_chunks and isinstance(stored_chunks, dict):
                existing_full_source_ids = [
                    chunk_id
                    for chunk_id in stored_chunks.get("chunk_ids", [])
                    if chunk_id
                ]

        if not existing_full_source_ids:
            existing_full_source_ids = [
                chunk_id for chunk_id in already_source_ids if chunk_id
            ]

        # 2. Merging new source ids with existing ones
        full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)
```

**File:** lightrag/operate.py (L2054-2064)
```python
        unique_nodes = {}
        for i, dp in enumerate(nodes_data, start=1):
            desc = dp.get("description")
            if not desc:
                continue
            if desc not in unique_nodes:
                unique_nodes[desc] = dp
            await _cooperative_yield(i, every=32)

        # Sort description by timestamp, then by description length when timestamps are the same
        sorted_nodes = sorted(
```

**File:** lightrag/operate.py (L2088-2095)
```python
        description, llm_was_used = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache,
        )
```

**File:** lightrag/operate.py (L2204-2207)
```python
        await knowledge_graph_inst.upsert_node(
            entity_name,
            node_data=node_data,
        )
```

**File:** lightrag/operate.py (L2225-2228)
```python
            await safe_vdb_operation_with_exception(
                operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
                operation_name="entity_upsert",
                entity_name=entity_name,
```

**File:** lightrag/operate.py (L2271-2272)
```python
        if await knowledge_graph_inst.has_edge(src_id, tgt_id):
            already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
```

**File:** lightrag/operate.py (L2390-2390)
```python
        weight = sum([dp["weight"] for dp in edges_data] + already_weights)
```

**File:** lightrag/operate.py (L2548-2554)
```python
        for need_insert_id in [src_id, tgt_id]:
            # Optimization: Use get_node instead of has_node + get_node
            existing_node = await knowledge_graph_inst.get_node(need_insert_id)

            if existing_node is None:
                # Node doesn't exist - create new node
                node_created_at = int(time.time())
```

**File:** lightrag/operate.py (L2826-2849)
```python
async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> None:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results
```

**File:** lightrag/operate.py (L2877-2888)
```python
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for i, (maybe_nodes, maybe_edges) in enumerate(chunk_results, start=1):
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)
```

**File:** lightrag/operate.py (L2911-2938)
```python
    async def _locked_process_entity_name(entity_name, entities):
        async with semaphore:
            # Check for cancellation before processing entity
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during entity merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    logger.debug(f"Processing entity {entity_name}")
                    entity_data = await _merge_nodes_then_upsert(
                        entity_name,
                        entities,
                        knowledge_graph_inst,
                        entity_vdb,
                        global_config,
                        pipeline_status,
                        pipeline_status_lock,
                        llm_response_cache,
                        entity_chunks_storage,
                    )
```

**File:** lightrag/operate.py (L2976-2978)
```python
        done, pending = await asyncio.wait(
            entity_tasks, return_when=asyncio.FIRST_EXCEPTION
        )
```

**File:** lightrag/operate.py (L3016-3039)
```python
    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            # Check for cancellation before processing edges
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during relation merge"
                        )

            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])
            edge_label = _format_relation_edge_label(edge_key)

            async with get_storage_keyed_lock(
                sorted_edge_key,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    added_entities = []  # Track entities added during edge processing

                    edge_data = await _merge_edges_then_upsert(
```

**File:** lightrag/operate.py (L3157-3165)
```python
    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            final_entity_names = set()

            # Add original processed entities
            for i, entity_data in enumerate(processed_entities, start=1):
                if entity_data and entity_data.get("entity_name"):
```

**File:** lightrag/operate.py (L3232-3235)
```python
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
```

**File:** lightrag/operate.py (L3334-3334)
```python
        content = strip_internal_multimodal_markup_for_extraction(chunk_dp["content"])
```

**File:** lightrag/operate.py (L3364-3374)
```python
        final_result, timestamp = await use_llm_func_with_cache(
            entity_extraction_user_prompt,
            use_llm_func,
            system_prompt=entity_extraction_system_prompt,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
            response_format=({"type": "json_object"} if use_json_extraction else None),
            llm_cache_identity=get_llm_cache_identity(global_config, "extract"),
        )
```

**File:** lightrag/operate.py (L3381-3396)
```python
        if use_json_extraction:
            maybe_nodes, maybe_edges = await _process_json_extraction_result(
                final_result,
                chunk_key,
                timestamp,
                file_path,
            )
        else:
            maybe_nodes, maybe_edges = await _process_extraction_result(
                final_result,
                chunk_key,
                timestamp,
                file_path,
                tuple_delimiter=context_base["tuple_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
            )
```

**File:** lightrag/operate.py (L3399-3420)
```python
        run_gleaning = entity_extract_max_gleaning > 0
        if (
            run_gleaning
            and extract_tokenizer is not None
            and max_extract_input_tokens > 0
        ):
            # Gleaning replays the initial extraction's user/assistant pair
            # via ``history_messages`` and appends a "continue" instruction.
            # When the initial response was large (many entities/edges) or
            # the chunk content is itself near the budget, that combined
            # payload can blow past MAX_EXTRACT_INPUT_TOKENS and yield a
            # provider ``context_length_exceeded`` error.  Pre-check here
            # and skip rather than fail.
            gleaning_token_count = (
                len(extract_tokenizer.encode(entity_extraction_system_prompt))
                + sum(
                    len(extract_tokenizer.encode(msg.get("content", "") or ""))
                    for msg in history
                )
                + len(extract_tokenizer.encode(entity_continue_extraction_user_prompt))
            )
            if gleaning_token_count > max_extract_input_tokens:
```

**File:** env.example (L425-439)
```text
### Per-response cap on total entity+relationship rows/records emitted by the LLM
# MAX_EXTRACTION_RECORDS=100
### Per-response cap on entity rows/objects emitted by the LLM
# MAX_EXTRACTION_ENTITIES=40

### Enable JSON-structured output for entity extraction (Note: JSON output incurs higher latency but delivers improved reliability)
### Default behavior: JSON output is disabled when ENTITY_EXTRACTION_USE_JSON is unset
ENTITY_EXTRACTION_USE_JSON=true

### Optional external YAML profile for entity type guidance and extraction examples
### Profiles are loaded from PROMPT_DIR/entity_type (PROMPT_DIR defaults to ./prompts).
### A reference template is shipped at prompts/samples/entity_type_prompt.sample.yml;
### Alternatively, override guidance at runtime from Python:
###   addon_params={"entity_types_guidance": "- CustomType: description..."}
# ENTITY_TYPE_PROMPT_FILE=entity_type_prompt.yml
```
