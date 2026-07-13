# TopoRAG
TopoRAG is a production-oriented Retrieval-Augmented Generation (RAG) system specialized in Topology, Differential Geometry, Algebraic Topology, and related mathematical fields. The project demonstrates modern LLM engineering practices, including embeddings, vector search, hybrid retrieval, reranking, prompt engineering, evaluation, observability, guardrails, fine-tuning, and GPU-accelerated inference.


## Notebook 01 — arXiv Corpus Collection

`01_arxiv_corpus_collection.ipynb` builds the initial scientific corpus used by TopoRAG.

The notebook retrieves research papers from arXiv related to differential geometry, algebraic topology, homology, cohomology, Topological Data Analysis, and persistent homology. It uses the official arXiv Python client directly, without LangChain or LangGraph, to keep the collection stage lightweight, reproducible, and independent from downstream RAG dependencies.

The collection pipeline performs the following steps:

1. Executes topic-specific arXiv search queries.
2. Retrieves paper metadata, including title, abstract, authors, categories, publication dates, and PDF URLs.
3. Normalizes and deduplicates papers using the arXiv identifier.
4. Preserves all associated topics when the same paper is returned by multiple queries.
5. Creates corpus manifests in CSV and Parquet formats.
6. Downloads the source PDFs while skipping files that already exist locally.
7. Validates downloaded PDF files before processing them.
8. Extracts raw text independently from every PDF page using PyMuPDF.
9. Joins page-level text with the corresponding paper metadata.
10. Runs structural and consistency validations.
11. Exports a page-level intermediate dataset in Parquet format.

### Outputs

```text
data/manifests/arxiv_corpus_manifest.csv
data/manifests/arxiv_corpus_manifest.parquet
data/raw/arxiv/*.pdf
data/interim/arxiv_raw_pages.parquet
```

The generated `arxiv_raw_pages.parquet` file is intentionally an intermediate dataset. It contains raw page-level text and metadata but does not perform text cleaning, Unicode normalization, header or footer removal, chunking, embedding generation, vector indexing, or retrieval.

Those transformations are handled by later notebooks in the TopoRAG pipeline.

### Separation of responsibilities

This notebook is responsible only for corpus acquisition and raw text extraction.

```text
arXiv queries
    ↓
Metadata retrieval
    ↓
Deduplication and topic aggregation
    ↓
Corpus manifest creation
    ↓
PDF download and validation
    ↓
Page-level text extraction
    ↓
Metadata enrichment
    ↓
Intermediate Parquet dataset
```

The next notebook, `02_document_ingestion.ipynb`, consumes the intermediate Parquet dataset and performs document cleaning, Unicode normalization, mathematical symbol preservation, repeated header and footer removal, low-quality page detection, and stable identifier generation.
