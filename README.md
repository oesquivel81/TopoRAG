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
# 02 — Scientific Document Ingestion and Cleaning

## Overview

This notebook prepares a corpus of scientific articles collected from arXiv for use in a Retrieval-Augmented Generation (RAG) system.

Its main purpose is to transform raw PDF extraction results into a corpus that is:

- Clean.
- Structured.
- Traceable.
- Free of empty pages.
- Free of exact duplicates.
- Able to preserve mathematical notation.
- Ready for chunking, embedding generation, and semantic retrieval.

The corpus focuses on the following domains:

- Differential geometry.
- Algebraic topology.
- Topological data analysis.
- Homology and cohomology.
- Persistent homology.

---

## Notebook Objective

The objective of this notebook is to build a reliable corpus in which every page:

- Preserves its original extracted text.
- Has a cleaned representation.
- Preserves mathematical symbols.
- Can be traced back to its source article.
- Retains its original page number.
- Has a preliminary quality label.
- Can be audited.
- Can be used by the next stages of the RAG pipeline.

This notebook does not generate embeddings or build a vector database.

Its responsibility is to prepare and validate the data before those operations are performed.

---

## Position in the RAG Pipeline

The complete system follows this workflow:

```text
arXiv search
    ↓
PDF download
    ↓
Page-level text extraction
    ↓
Corpus inspection
    ↓
Conservative text cleaning
    ↓
Header and footer removal
    ↓
Page quality classification
    ↓
Exact duplicate detection
    ↓
Clean page-level corpus
    ↓
Chunking
    ↓
Embedding generation
    ↓
Vector store
    ↓
Retrieval
    ↓
Answer generation
    ↓
Evaluation
```

This notebook covers the stages between arXiv collection and clean corpus creation.

---

## Why Is Corpus Cleaning Necessary?

Text extracted directly from PDF files is not always suitable for a RAG system.

Common extraction problems include:

- Repeated headers.
- Repeated footers.
- Page numbers.
- Repeated document titles.
- Author names on every page.
- Excessive whitespace.
- Invisible control characters.
- Empty pages.
- Duplicate content.
- Words divided across line endings.
- Pages containing mostly images.
- Extraction errors.

For example, a word may be extracted as:

```text
homo-
logy
```

Its correct representation should be:

```text
homology
```

If these artifacts are embedded directly, they also become part of the vector index.

This can lead to:

- Lower retrieval precision.
- Irrelevant document retrieval.
- Repeated context.
- Incorrect citations.
- Increased hallucination risk.

Cleaning is therefore not merely cosmetic. It is a retrieval-quality and data-engineering stage.

---

## Data Source

The corpus is collected from arXiv using:

```python
ArxivCorpusCollector
```

This component is responsible for:

1. Running arXiv search queries.
2. Retrieving article metadata.
3. Downloading PDF documents.
4. Extracting text page by page.
5. Creating the `corpus_df` DataFrame.

---

## Search Domains

The corpus is initially organized into three domains.

### Differential Geometry

```text
Differential geometry
Riemannian geometry
```

### Algebraic Topology

```text
Algebraic topology
Homology
Cohomology
```

### Topological Data Analysis

```text
Persistent homology
Topological data analysis
```

The search queries combine arXiv categories with domain-specific terms.

---

## Corpus Structure

Each row in `corpus_df` represents one extracted PDF page.

The DataFrame does not contain one row per article. It contains one row per page.

Example:

```text
Article A, page 1 → row 1
Article A, page 2 → row 2
Article A, page 3 → row 3
Article B, page 1 → row 4
```

The current execution produced approximately:

```text
1,325 pages
37 columns
```

The exact values may change depending on the search configuration and article limits.

---

## Main Columns

| Column | Description |
|---|---|
| `arxiv_id` | Stable arXiv article identifier |
| `source_file` | Source PDF filename |
| `page_number` | Page number within the document |
| `total_pages` | Total number of pages |
| `page_content` | Extracted page text |
| `char_count` | Number of extracted characters |
| `word_count` | Number of extracted words |
| `is_empty_raw` | Indicates whether the extracted page is empty |
| `title` | Article title |
| `authors` | Article authors |
| `published` | Publication date |
| `updated` | Last arXiv update date |
| `primary_category` | Primary arXiv category |
| `categories` | Associated arXiv categories |
| `summary` | Article abstract |
| `topics` | Topics used to retrieve the document |
| `pdf_url` | Original PDF URL |
| `download_status` | PDF download status |
| `extraction_status` | PDF extraction status |
| `extraction_error` | Extraction error, when applicable |
| `download_error` | Download error, when applicable |

These metadata fields allow the RAG system to generate references containing:

```text
Article title
arXiv ID
Page number
Source URL
```

---

# Notebook Methodology

## 1. Environment Setup

The notebook installs the dependencies required for:

- arXiv collection.
- PDF extraction.
- DataFrame processing.
- Text cleaning.
- Future LangChain integration.

The main libraries include:

```text
PyMuPDF
Pandas
LangChain Core
LangChain Community
LangChain OpenAI
```

The TopoRAG repository is also loaded because it contains the reusable corpus-collection implementation.

### Pandas Compatibility

Google Colab may require a specific Pandas version.

The compatible version should be installed before importing Pandas and before running the collector.

Recommended order:

```text
Install dependencies
    ↓
Install the compatible Pandas version
    ↓
Restart the runtime when required
    ↓
Import libraries
    ↓
Run the collector
```

---

## 2. Repository Setup

The repository should only be cloned when it is not already available in the current Colab session.

Example:

```python
from pathlib import Path
import subprocess

PROJECT_ROOT = Path("/content/TopoRAG")

if not (PROJECT_ROOT / ".git").exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/oesquivel81/TopoRAG.git",
            str(PROJECT_ROOT),
        ],
        check=True,
    )
else:
    print("The repository already exists:", PROJECT_ROOT)
```

This makes the notebook idempotent and prevents repeated cloning errors.

---

## 3. Scientific Corpus Collection

The collector executes the configured arXiv queries and builds the corpus.

The internal workflow is:

```text
arXiv query
    ↓
Metadata retrieval
    ↓
PDF download
    ↓
Page-level extraction
    ↓
corpus_df creation
```

The main result is:

```python
corpus_df = result["corpus_df"]
```

---

## 4. Corpus Structure Inspection

Before modifying any content, the notebook inspects:

- DataFrame dimensions.
- Column names.
- Data types.
- Missing values.
- Initial rows.
- Memory usage.

Example:

```python
display(corpus_df.head())
corpus_df.info()
```

This verifies that the ingestion stage completed successfully.

---

## 5. Missing-Value Analysis

The notebook creates a report of missing values.

Not every missing value represents an error.

For example:

```text
download_error = null
```

usually means that no download exception occurred.

Similarly:

```text
extraction_error = null
```

usually means that extraction completed without an exception.

The following metadata fields are optional:

```text
doi
journal_ref
comment
```

Many arXiv articles do not contain them.

---

## 6. Corpus-Level Statistics

The notebook calculates:

- Total pages.
- Unique articles.
- Unique PDF files.
- Empty pages.
- Total characters.
- Total words.

Example:

```python
corpus_summary = {
    "rows_pages": len(corpus_df),
    "unique_articles": corpus_df["arxiv_id"].nunique(),
    "unique_pdf_files": corpus_df["source_file"].nunique(),
    "empty_pages": int(corpus_df["is_empty_raw"].sum()),
    "total_characters": int(corpus_df["char_count"].sum()),
    "total_words": int(corpus_df["word_count"].sum()),
}
```

This summary establishes a baseline for comparing the raw and cleaned corpora.

---

## 7. Article-Level Validation

Page records are grouped by:

```text
arxiv_id
title
source_file
```

For each article, the notebook calculates:

- Number of extracted pages.
- Number of pages reported by the PDF.
- Total characters.
- Total words.
- Associated topics.

This helps detect incomplete extractions.

Example:

```text
Reported pages: 20
Extracted pages: 20
Missing pages: 0
```

If the extracted page count is lower than the reported page count, the article should be inspected.

---

## 8. Download and Extraction Validation

The notebook reviews:

```text
download_status
extraction_status
```

It also identifies pages marked as empty:

```python
corpus_df["is_empty_raw"]
```

An empty page may represent:

- A blank page.
- A cover page.
- A full-page figure.
- A diagram.
- Scanned content.
- A page requiring OCR.
- An extraction failure.

Empty pages are reviewed before removal.

---

## 9. Original Corpus Preservation

The original DataFrame is not modified directly.

A working copy is created:

```python
working_df = corpus_df.copy()
```

The original extracted content is also preserved:

```python
working_df["raw_text"] = (
    working_df["page_content"]
    .fillna("")
    .astype(str)
)
```

The conceptual structure is:

```text
corpus_df
    └── original page-level corpus

working_df
    ├── raw_text
    ├── basic_clean_text
    └── clean_text
```

This makes every transformation auditable.

---

## 10. Conservative Mathematical Text Cleaning

The main cleaning function is:

```python
clean_mathematical_text()
```

The function applies limited and controlled transformations.

Its goal is to remove extraction artifacts without destroying mathematical notation.

Symbols such as the following must be preserved:

```text
π
∂
ℝ
Hₙ
⊗
∫
≤
≥
```

The cleaning process performs the following operations.

### 10.1 Unicode Normalization

```python
unicodedata.normalize("NFC", text)
```

NFC normalization provides a consistent Unicode representation without aggressively transforming mathematical characters.

### 10.2 Null Byte Removal

```python
text = text.replace("\x00", "")
```

Null bytes may appear as PDF extraction artifacts.

### 10.3 Control Character Removal

Invisible control characters are removed while preserving:

```text
Line breaks
Tabs
```

This retains a minimal document structure.

### 10.4 Hyphenated Word Repair

Example:

```text
cohomo-
logy
```

becomes:

```text
cohomology
```

The repair is applied only when the word appears to have been divided by PDF line wrapping.

### 10.5 Whitespace Normalization

Repeated spaces and tabs are reduced.

Example:

```text
persistent      homology
```

becomes:

```text
persistent homology
```

### 10.6 Line-Break Normalization

Unnecessary spaces around line breaks are removed, and excessive blank lines are reduced.

Paragraph boundaries are retained.

### 10.7 Punctuation Correction

Example:

```text
topology ,
```

becomes:

```text
topology,
```

---

## 11. Applying Basic Cleaning

The cleaning function is applied to every page:

```python
working_df["basic_clean_text"] = (
    working_df["raw_text"]
    .apply(clean_mathematical_text)
)
```

The notebook also recalculates:

- Clean character count.
- Clean word count.

The original representation remains available.

---

## 12. Before-and-After Inspection

Selected pages are displayed before and after cleaning.

This review verifies that:

- Mathematical notation remains intact.
- Divided words are repaired correctly.
- Paragraphs remain readable.
- Definitions are preserved.
- Theorems are preserved.
- Equations are not accidentally removed.

Automated metrics are not sufficient for validating mathematical text.

---

## 13. Repeated Header and Footer Detection

Scientific PDFs often repeat lines such as:

```text
Article title
Author names
Journal name
arXiv ID
Page number
```

These repeated lines add noise to the vector index.

Detection is performed separately for every article using:

```python
arxiv_id
```

This is important because every PDF may use a different layout.

---

## 14. Boundary-Line Normalization

Helper functions are used to:

- Obtain non-empty lines.
- Normalize whitespace.
- Convert lines to lowercase.
- Identify standalone page numbers.

Examples of standalone page numbers include:

```text
14
```

and:

```text
Page 14
```

---

## 15. Frequency-Based Boundary Detection

The notebook examines the first and last lines of every page.

The initial configuration is:

```python
boundary_depth = 2
minimum_ratio = 0.30
```

This means:

- The first two lines are inspected.
- The last two lines are inspected.
- A line must appear on at least 30% of the article’s pages.
- A line must appear at least three times.

Example:

```text
Article with 20 pages
30% of 20 = 6 pages
```

A line appearing on six or more pages may be classified as a header or footer candidate.

---

## 16. Header and Footer Review

Detected candidates are displayed before removal.

Example:

```text
Headers:
- algebraic topology
- john smith and jane doe

Footers:
- arxiv:2601.00001
```

A repeated line is not always noise.

It may also be:

- A section title.
- A mathematical expression.
- A theorem label.
- Legitimate article content.

The correct methodology is:

```text
Detect
    ↓
Review
    ↓
Remove
```

Content should not be removed automatically without inspection.

---

## 17. Header, Footer, and Page-Number Removal

The boundary-removal function only examines the first and last lines of each page.

It may remove:

- Repeated headers.
- Repeated footers.
- Standalone page numbers.

It does not inspect the central body of the page.

This reduces the risk of deleting legitimate scientific content.

The result is stored in:

```python
working_df["clean_text"]
```

---

## 18. Cleaning Impact Analysis

The notebook calculates:

```text
Original characters
Clean characters
Removed characters
Removed percentage
```

Example:

```text
Original characters: 2,000
Clean characters:    1,900
Removed characters:   100
Removed percentage:     5%
```

A small reduction is expected.

Pages that lose more than 25% of their original content are flagged for review.

A high percentage may correspond to:

- A cover page.
- A nearly empty page.
- A page with extensive repeated metadata.
- A figure page.
- An overly aggressive cleaning rule.

---

## 19. Page Quality Classification

Every page receives a preliminary quality label.

| Classification | Rule |
|---|---|
| `empty` | No text remains |
| `suspicious_short` | Fewer than 100 characters or 15 words |
| `short` | Fewer than 300 characters or 40 words |
| `usable` | Contains enough text for normal processing |

This classification describes extraction quality.

It does not measure the scientific quality of an article.

A short page may still contain:

- A definition.
- An equation.
- A proposition.
- A figure caption.
- A reference.
- A short conclusion.

For that reason, short pages are not automatically removed.

---

## 20. Suspicious Page Review

Pages classified as:

```text
empty
suspicious_short
```

are inspected.

This helps determine whether they contain:

- Blank content.
- Figures.
- Scanned text.
- Equations.
- Extraction failures.
- Content requiring OCR.

OCR is outside the current notebook’s scope.

---

## 21. Exact Duplicate Detection

Each cleaned page receives a SHA-256 hash.

The process is:

```text
Clean text
    ↓
Whitespace normalization
    ↓
Lowercase conversion
    ↓
SHA-256
    ↓
Content hash
```

If two pages have the same hash, they contain exactly the same normalized text.

Duplicates may appear because:

- Different searches retrieved the same article.
- The same PDF was downloaded multiple times.
- A document contains repeated pages.
- One article belongs to several topics.

This notebook detects exact duplicates only.

It does not perform semantic duplicate detection.

---

## 22. Final Clean Corpus Construction

The final DataFrame is:

```python
clean_corpus_df
```

Only the following records are removed automatically:

- Completely empty pages.
- Exact duplicate pages.

Short pages remain available for later review.

The original extracted text is stored in:

```python
raw_page_content
```

The cleaned version becomes the canonical content field:

```python
page_content
```

This allows the next notebook to consume a consistent column name.

---

# Generated Files

The results are stored in:

```text
/content/TopoRAG/data/processed/
```

## `arxiv_pages_clean.parquet`

The primary analytical dataset.

Advantages:

- Preserves data types.
- Loads efficiently.
- Uses less storage.
- Works well with Pandas.
- Preserves lists and dates better than CSV.

## `arxiv_pages_clean.jsonl`

Contains one JSON object per line.

It is useful for:

- Document pipelines.
- Batch processing.
- LangChain integration.
- Sequential reading.
- Data exchange between systems.

## `page_quality_report.csv`

The human-readable quality-control report.

It contains fields such as:

```text
arxiv_id
title
source_file
page_number
total_pages
char_count
word_count
clean_char_count
clean_word_count
removed_percentage
quality
is_exact_duplicate
content_hash
```

This report allows the corpus to be reviewed without loading all full-text content.

---

# Data Flow

```text
Original PDFs
    ↓
ArxivCorpusCollector
    ↓
corpus_df
    ↓
working_df
    ├── raw_text
    ├── basic_clean_text
    ├── clean_text
    ├── quality
    ├── content_hash
    └── is_exact_duplicate
    ↓
clean_corpus_df
    ↓
Parquet
JSONL
CSV quality report
```

---

# Success Criteria

The notebook is considered successful when:

- Articles are downloaded correctly.
- PDF pages are extracted.
- Metadata is preserved.
- Original text remains available.
- Mathematical notation is preserved.
- Repeated headers and footers are reduced.
- Empty pages are identified.
- Suspicious pages can be reviewed.
- Exact duplicates are detected.
- Every page can be traced to its source article.
- The clean corpus can be loaded by the next notebook.

---

# What This Notebook Does Not Perform

This notebook does not yet perform:

- Token-aware chunking.
- Embedding generation.
- FAISS index creation.
- Semantic search.
- Hybrid search.
- Reranking.
- Answer generation.
- Complete LLM integration.
- Retrieval evaluation.
- LangSmith tracing.
- OCR for scanned pages.
- Semantic duplicate detection.

These stages remain separate so that data quality can be validated before embeddings are generated.

---

# Final Result

The complete transformation performed in this notebook is:

```text
arXiv articles
    ↓
PDF download
    ↓
Page-level extraction
    ↓
Corpus inspection
    ↓
Original text preservation
    ↓
Conservative mathematical cleaning
    ↓
Header and footer detection
    ↓
Page-number removal
    ↓
Cleaning impact measurement
    ↓
Page quality classification
    ↓
Exact duplicate detection
    ↓
Clean corpus
    ↓
Parquet, JSONL, and CSV export
```

---

# Importance in a RAG Project

RAG quality depends directly on corpus quality.

An incorrect pipeline would be:

```text
Unreviewed PDF content
    ↓
Embeddings
    ↓
Vector store
```

The correct methodology is:

```text
PDF
    ↓
Extraction
    ↓
Inspection
    ↓
Cleaning
    ↓
Quality control
    ↓
Chunking
    ↓
Embeddings
```

Embeddings should not be generated until the corpus has been reviewed and validated.

---

# Skills Demonstrated

This notebook demonstrates experience in:

- Document ingestion.
- PDF processing.
- RAG data preparation.
- Reproducible data pipelines.
- Data-quality analysis.
- Text normalization.
- Scientific document processing.
- Mathematical notation preservation.
- Document traceability.
- Exact deduplication.
- Metadata management.
- Semantic-search preparation.

Corpus cleaning represents the **data engineering and knowledge-preparation stage** of the RAG system.

---

# Next Stage

The next notebook will consume:

```text
arxiv_pages_clean.parquet
```

Recommended notebook name:

```text
03_chunking_embeddings_and_vector_store.ipynb
```

Its workflow will be:

```text
Clean corpus
    ↓
Token-aware chunking
    ↓
Metadata propagation
    ↓
Chunk identifiers
    ↓
Embedding generation
    ↓
FAISS vector index
    ↓
Retrieval testing
```

The main principle for the next stage is:

> Evaluate retrieval quality before connecting the final LLM generation layer.

Prompt engineering cannot compensate for a retrieval system that returns incorrect documents.
