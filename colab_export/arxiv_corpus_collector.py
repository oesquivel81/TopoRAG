from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Final, Iterable

import arxiv
import fitz
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_SEARCH_QUERIES: Final[dict[str, str]] = {
    "differential_geometry": (
        'cat:math.DG AND '
        '(all:"differential geometry" OR all:"Riemannian geometry")'
    ),
    "algebraic_topology": (
        'cat:math.AT AND '
        '(all:"algebraic topology" OR all:"homology" OR all:"cohomology")'
    ),
    "topological_data_analysis": (
        '(cat:math.AT OR cat:cs.LG OR cat:stat.ML) AND '
        '(all:"persistent homology" OR all:"topological data analysis")'
    ),
}


class ArxivCorpusCollector:
    """Clase reutilizable para construir un corpus arXiv y dejarlo consolidado."""

    def __init__(
        self,
        search_queries: dict[str, str] | str | None = None,
        project_root: str | Path | None = None,
        max_results_per_query: int = 10,
    ) -> None:
        self.project_root = Path(project_root or "/content/TopoRAG").resolve()
        self.search_queries = self._normalize_search_queries(search_queries)
        self.max_results_per_query = max_results_per_query

        self.raw_pdf_dir = self.project_root / "data" / "raw" / "arxiv"
        self.manifest_dir = self.project_root / "data" / "manifests"
        self.interim_dir = self.project_root / "data" / "interim"
        self.processed_dir = self.project_root / "data" / "processed"

        self.manifest_csv_path = self.manifest_dir / "arxiv_corpus_manifest.csv"
        self.manifest_parquet_path = self.manifest_dir / "arxiv_corpus_manifest.parquet"
        self.raw_pages_parquet_path = self.interim_dir / "arxiv_raw_pages.parquet"

        self.logger = self._configure_logger()
        self._prepare_directories()

        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=3,
            num_retries=3,
        )

        self.query_hits_df: pd.DataFrame | None = None
        self.query_errors_df: pd.DataFrame | None = None
        self.manifest_df: pd.DataFrame | None = None
        self.pages_df: pd.DataFrame | None = None
        self.corpus_df: pd.DataFrame | None = None
        self.validation_df: pd.DataFrame | None = None

    @staticmethod
    def _configure_logger() -> logging.Logger:
        logger = logging.getLogger("toporag.arxiv_collection")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            logger.addHandler(handler)

        return logger

    def _normalize_search_queries(self, search_queries: dict[str, str] | str | None) -> dict[str, str]:
        if search_queries is None:
            return dict(DEFAULT_SEARCH_QUERIES)

        if isinstance(search_queries, str):
            try:
                normalized = json.loads(search_queries)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido para search_queries: {exc}") from exc
            if not isinstance(normalized, dict):
                raise TypeError("search_queries debe ser un dict o un JSON válido")
            return {str(k): str(v) for k, v in normalized.items()}

        if isinstance(search_queries, dict):
            return {str(k): str(v) for k, v in search_queries.items()}

        raise TypeError("search_queries debe ser un dict, un JSON o None")

    def _prepare_directories(self) -> None:
        for directory in (
            self.raw_pdf_dir,
            self.manifest_dir,
            self.interim_dir,
            self.processed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Project root: %s", self.project_root)
        self.logger.info("Raw PDF directory: %s", self.raw_pdf_dir)
        self.logger.info("Manifest directory: %s", self.manifest_dir)
        self.logger.info("Interim directory: %s", self.interim_dir)

    def collect_query_hits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.max_results_per_query <= 0:
            raise ValueError("max_results_per_query debe ser positivo")

        hit_records: list[dict[str, Any]] = []
        error_records: list[dict[str, str]] = []

        for topic, query in self.search_queries.items():
            self.logger.info("Searching topic: %s", topic)

            search = arxiv.Search(
                query=query,
                max_results=self.max_results_per_query,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            topic_count = 0
            try:
                for paper in self.client.results(search):
                    hit_records.append(self._result_to_hit_record(paper, topic, query))
                    topic_count += 1

                self.logger.info("Retrieved %d records for topic: %s", topic_count, topic)
            except Exception as exc:
                self.logger.exception("Metadata retrieval failed for topic: %s", topic)
                error_records.append(
                    {
                        "topic": topic,
                        "search_query": query,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        hits_df = pd.DataFrame(hit_records)
        errors_df = pd.DataFrame(
            error_records,
            columns=["topic", "search_query", "error_type", "error_message"],
        )

        if hits_df.empty:
            raise RuntimeError("No arXiv metadata was retrieved.")

        self.query_hits_df = hits_df
        self.query_errors_df = errors_df
        return hits_df, errors_df

    @staticmethod
    def _result_to_hit_record(paper: arxiv.Result, topic: str, search_query: str) -> dict[str, Any]:
        short_id = paper.get_short_id()
        arxiv_id, arxiv_version = ArxivCorpusCollector._split_arxiv_identifier(short_id)

        return {
            "arxiv_id": arxiv_id,
            "arxiv_version": arxiv_version,
            "entry_id": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "title": ArxivCorpusCollector._normalize_metadata_text(paper.title),
            "authors": [
                author.name.strip() for author in paper.authors if author.name.strip()
            ],
            "published": paper.published,
            "updated": paper.updated,
            "primary_category": paper.primary_category,
            "categories": sorted(set(paper.categories)),
            "summary": ArxivCorpusCollector._normalize_metadata_text(paper.summary),
            "comment": ArxivCorpusCollector._normalize_metadata_text(paper.comment),
            "journal_ref": ArxivCorpusCollector._normalize_metadata_text(paper.journal_ref),
            "doi": ArxivCorpusCollector._normalize_metadata_text(paper.doi),
            "topic": topic,
            "search_query": search_query,
            "source_type": "arxiv",
            "retrieved_at_utc": pd.Timestamp.now(tz="UTC"),
        }

    @staticmethod
    def _split_arxiv_identifier(short_id: str) -> tuple[str, int | None]:
        normalized = short_id.strip()
        if not normalized:
            raise ValueError("El identificador arXiv no puede estar vacío")

        version_match = re.search(r"v(?P<version>\d+)$", normalized)
        if version_match is None:
            return normalized, None

        version = int(version_match.group("version"))
        canonical_id = re.sub(r"v\d+$", "", normalized)
        return canonical_id, version

    @staticmethod
    def _normalize_metadata_text(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def build_corpus_manifest(self, query_hits: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"arxiv_id", "updated", "topic", "search_query"}
        missing_columns = required_columns.difference(query_hits.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

        if query_hits.empty:
            raise ValueError("No se puede construir un manifiesto desde un DataFrame vacío")

        latest_metadata = (
            query_hits.sort_values(
                by=["arxiv_id", "updated"],
                ascending=[True, True],
                na_position="first",
            )
            .drop_duplicates(subset=["arxiv_id"], keep="last")
            .drop(columns=["topic", "search_query"])
        )

        provenance = (
            query_hits.groupby("arxiv_id", as_index=False)
            .agg(
                topics=("topic", self._sorted_unique_strings),
                search_queries=("search_query", self._sorted_unique_strings),
                topic_count=("topic", lambda values: len(self._sorted_unique_strings(values))),
                query_hit_count=("topic", "size"),
            )
        )

        manifest = latest_metadata.merge(provenance, on="arxiv_id", how="inner", validate="one_to_one")
        manifest["pdf_filename"] = manifest["arxiv_id"].map(self._make_pdf_filename)
        manifest["pdf_relative_path"] = "data/raw/arxiv/" + manifest["pdf_filename"]
        manifest["download_status"] = "pending"
        manifest["download_error"] = None
        manifest["pdf_size_bytes"] = pd.Series([pd.NA] * len(manifest), dtype="Int64")
        manifest["downloaded_at_utc"] = pd.NaT
        manifest["extraction_status"] = "pending"
        manifest["extraction_error"] = None
        manifest["extracted_page_count"] = pd.Series([pd.NA] * len(manifest), dtype="Int64")

        manifest = manifest.sort_values(
            by=["published", "arxiv_id"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

        if manifest["arxiv_id"].duplicated().any():
            raise RuntimeError("La deduplicación del manifiesto falló")

        self.manifest_df = manifest
        return manifest

    @staticmethod
    def _sorted_unique_strings(values: Iterable[str]) -> list[str]:
        return sorted({value.strip() for value in values if isinstance(value, str) and value.strip()})

    @staticmethod
    def _make_pdf_filename(arxiv_id: str) -> str:
        return f"{arxiv_id.replace('/', '__')}.pdf"

    def export_manifest(self, manifest: pd.DataFrame | None = None) -> None:
        manifest_to_export = manifest if manifest is not None else self.manifest_df
        if manifest_to_export is None:
            raise RuntimeError("No hay manifiesto para exportar")

        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

        manifest_to_export.to_parquet(
            self.manifest_parquet_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        csv_manifest = manifest_to_export.copy()
        for column in ("authors", "categories", "topics", "search_queries"):
            if column in csv_manifest.columns:
                csv_manifest[column] = csv_manifest[column].map(
                    lambda value: json.dumps(value if isinstance(value, list) else [], ensure_ascii=False)
                )

        csv_manifest.to_csv(self.manifest_csv_path, index=False, encoding="utf-8")

    def create_download_session(self) -> requests.Session:
        retry_policy = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_policy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": "TopoRAG/0.1 (research corpus collection; Python requests)"
        })
        return session

    def validate_pdf_file(self, pdf_path: Path) -> tuple[bool, str | None]:
        if not pdf_path.exists():
            return False, "File does not exist."
        if pdf_path.stat().st_size == 0:
            return False, "File is empty."
        try:
            with fitz.open(str(pdf_path)) as document:
                if document.page_count <= 0:
                    return False, "PDF contains no pages."
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def next_invalid_backup_path(self, pdf_path: Path) -> Path:
        candidate = pdf_path.with_suffix(".invalid.pdf")
        counter = 1
        while candidate.exists():
            candidate = pdf_path.with_suffix(f".invalid.{counter}.pdf")
            counter += 1
        return candidate

    def download_pdf(self, session: requests.Session, arxiv_id: str, pdf_url: str | None, output_path: Path) -> dict[str, Any]:
        base_record: dict[str, Any] = {
            "arxiv_id": arxiv_id,
            "download_status": "failed",
            "download_error": None,
            "pdf_size_bytes": pd.NA,
            "downloaded_at_utc": pd.NaT,
        }

        if output_path.exists():
            is_valid, validation_error = self.validate_pdf_file(output_path)
            if is_valid:
                return {**base_record, "download_status": "existing", "pdf_size_bytes": output_path.stat().st_size}

            backup_path = self.next_invalid_backup_path(output_path)
            output_path.replace(backup_path)
            self.logger.warning("Preserved invalid existing PDF as %s: %s", backup_path.name, validation_error)

        if not pdf_url:
            return {**base_record, "download_status": "missing_pdf_url", "download_error": "arXiv metadata did not provide a PDF URL."}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".pdf.part")

        try:
            with session.get(pdf_url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)

            is_valid, validation_error = self.validate_pdf_file(temporary_path)
            if not is_valid:
                raise ValueError(f"Downloaded file is not a valid PDF: {validation_error}")

            temporary_path.replace(output_path)
            return {
                **base_record,
                "download_status": "downloaded",
                "pdf_size_bytes": output_path.stat().st_size,
                "downloaded_at_utc": pd.Timestamp.now(tz="UTC"),
            }
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            return {**base_record, "download_status": "failed", "download_error": f"{type(exc).__name__}: {exc}"}

    def download_manifest_pdfs(self, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
        manifest_to_use = manifest if manifest is not None else self.manifest_df
        if manifest_to_use is None:
            raise RuntimeError("No hay manifiesto para descargar")

        result_records: list[dict[str, Any]] = []
        session = self.create_download_session()
        try:
            for row in manifest_to_use.itertuples(index=False):
                output_path = self.raw_pdf_dir / row.pdf_filename
                result = self.download_pdf(
                    session=session,
                    arxiv_id=row.arxiv_id,
                    pdf_url=row.pdf_url,
                    output_path=output_path,
                )
                result_records.append(result)
                self.logger.info("PDF %s: %s", row.arxiv_id, result["download_status"])
                if result["download_status"] == "downloaded":
                    time.sleep(1.0)
        finally:
            session.close()

        return pd.DataFrame(result_records)

    def extract_pdf_pages(self, pdf_path: Path, arxiv_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        page_records: list[dict[str, Any]] = []
        status_record: dict[str, Any] = {
            "arxiv_id": arxiv_id,
            "extraction_status": "failed",
            "extraction_error": None,
            "extracted_page_count": pd.NA,
        }

        try:
            with fitz.open(str(pdf_path)) as document:
                total_pages = document.page_count
                if total_pages <= 0:
                    raise ValueError("PDF contains no pages.")

                extraction_timestamp = pd.Timestamp.now(tz="UTC")
                for page_index in range(total_pages):
                    page = document.load_page(page_index)
                    raw_text = page.get_text("text")
                    page_records.append(
                        {
                            "arxiv_id": arxiv_id,
                            "source_file": pdf_path.name,
                            "page_number": page_index + 1,
                            "total_pages": total_pages,
                            "page_content": raw_text,
                            "char_count": len(raw_text),
                            "word_count": len(raw_text.split()),
                            "is_empty_raw": not bool(raw_text.strip()),
                            "extracted_at_utc": extraction_timestamp,
                        }
                    )

            status_record.update({
                "extraction_status": "success",
                "extracted_page_count": len(page_records),
            })
            return page_records, status_record
        except Exception as exc:
            status_record["extraction_error"] = f"{type(exc).__name__}: {exc}"
            return [], status_record

    def extract_manifest_pages(self, manifest: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        manifest_to_use = manifest if manifest is not None else self.manifest_df
        if manifest_to_use is None:
            raise RuntimeError("No hay manifiesto para extraer")

        all_page_records: list[dict[str, Any]] = []
        extraction_records: list[dict[str, Any]] = []
        available_statuses = {"downloaded", "existing"}

        for row in manifest_to_use.itertuples(index=False):
            if row.download_status not in available_statuses:
                extraction_records.append(
                    {
                        "arxiv_id": row.arxiv_id,
                        "extraction_status": "skipped",
                        "extraction_error": f"PDF unavailable because download status is {row.download_status}.",
                        "extracted_page_count": pd.NA,
                    }
                )
                continue

            pdf_path = self.raw_pdf_dir / row.pdf_filename
            page_records, extraction_status = self.extract_pdf_pages(pdf_path=pdf_path, arxiv_id=row.arxiv_id)
            all_page_records.extend(page_records)
            extraction_records.append(extraction_status)
            self.logger.info(
                "Extraction %s: %s (%s pages)",
                row.arxiv_id,
                extraction_status["extraction_status"],
                extraction_status["extracted_page_count"],
            )

        pages_df = pd.DataFrame(
            all_page_records,
            columns=[
                "arxiv_id",
                "source_file",
                "page_number",
                "total_pages",
                "page_content",
                "char_count",
                "word_count",
                "is_empty_raw",
                "extracted_at_utc",
            ],
        )
        extraction_df = pd.DataFrame(
            extraction_records,
            columns=["arxiv_id", "extraction_status", "extraction_error", "extracted_page_count"],
        )
        self.pages_df = pages_df
        return pages_df, extraction_df

    def validate_collection_outputs(self, manifest: pd.DataFrame | None = None, pages: pd.DataFrame | None = None, corpus: pd.DataFrame | None = None) -> pd.DataFrame:
        manifest_to_use = manifest if manifest is not None else self.manifest_df
        pages_to_use = pages if pages is not None else self.pages_df
        corpus_to_use = corpus if corpus is not None else self.corpus_df

        if manifest_to_use is None or pages_to_use is None or corpus_to_use is None:
            raise RuntimeError("Faltan datos para validar")

        validations: list[dict[str, Any]] = []

        validations.append(
            {
                "check_name": "manifest_not_empty",
                "severity": "error",
                "passed": not manifest_to_use.empty,
                "details": f"Manifest rows: {len(manifest_to_use)}",
            }
        )
        validations.append(
            {
                "check_name": "pages_not_empty",
                "severity": "error",
                "passed": not pages_to_use.empty,
                "details": f"Extracted page rows: {len(pages_to_use)}",
            }
        )
        validations.append(
            {
                "check_name": "manifest_unique_arxiv_id",
                "severity": "error",
                "passed": int(manifest_to_use["arxiv_id"].duplicated().sum()) == 0,
                "details": f"Duplicate IDs: {int(manifest_to_use['arxiv_id'].duplicated().sum())}",
            }
        )
        validations.append(
            {
                "check_name": "page_metadata_join_complete",
                "severity": "error",
                "passed": corpus_to_use["title"].notna().all() and corpus_to_use["topics"].notna().all(),
                "details": f"Pages missing titles: {int(corpus_to_use['title'].isna().sum())}; pages missing topics: {int(corpus_to_use['topics'].isna().sum())}",
            }
        )
        validations.append(
            {
                "check_name": "output_directory_separation",
                "severity": "error",
                "passed": self.manifest_csv_path.parent == self.manifest_dir and self.manifest_parquet_path.parent == self.manifest_dir and self.raw_pages_parquet_path.parent == self.interim_dir and self.raw_pages_parquet_path.parent != self.processed_dir,
                "details": "Manifests must remain in manifests and raw pages must remain in interim.",
            }
        )

        validation_df = pd.DataFrame(validations)
        self.validation_df = validation_df
        return validation_df

    def build_corpus(self) -> dict[str, pd.DataFrame]:
        query_hits_df, query_errors_df = self.collect_query_hits()
        manifest_df = self.build_corpus_manifest(query_hits_df)
        self.export_manifest(manifest_df)

        download_results_df = self.download_manifest_pdfs(manifest_df)
        download_columns = ["arxiv_id", "download_status", "download_error", "pdf_size_bytes", "downloaded_at_utc"]
        manifest_df = (
            manifest_df.drop(columns=download_columns[1:])
            .merge(download_results_df[download_columns], on="arxiv_id", how="left", validate="one_to_one")
        )

        pages_df, extraction_results_df = self.extract_manifest_pages(manifest_df)
        extraction_columns = ["arxiv_id", "extraction_status", "extraction_error", "extracted_page_count"]
        manifest_df = (
            manifest_df.drop(columns=extraction_columns[1:])
            .merge(extraction_results_df[extraction_columns], on="arxiv_id", how="left", validate="one_to_one")
        )

        if pages_df.empty:
            raise RuntimeError("No pages were extracted")

        corpus_df = pages_df.merge(manifest_df, on="arxiv_id", how="left", validate="many_to_one")
        self.manifest_df = manifest_df
        self.pages_df = pages_df
        self.corpus_df = corpus_df

        self.validation_df = self.validate_collection_outputs(manifest_df, pages_df, corpus_df)
        self._export_final_artifacts(manifest_df, corpus_df)

        self.query_hits_df = query_hits_df
        self.query_errors_df = query_errors_df
        return {
            "query_hits_df": query_hits_df,
            "query_errors_df": query_errors_df,
            "manifest_df": manifest_df,
            "pages_df": pages_df,
            "corpus_df": corpus_df,
            "validation_df": self.validation_df,
        }

    def _export_final_artifacts(self, manifest_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
        self.export_manifest(manifest_df)
        corpus_df.to_parquet(
            self.raw_pages_parquet_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

    def run(self) -> dict[str, pd.DataFrame]:
        return self.build_corpus()


if __name__ == "__main__":
    collector = ArxivCorpusCollector(
        search_queries=DEFAULT_SEARCH_QUERIES,
        project_root="/content/TopoRAG",
        max_results_per_query=10,
    )
    result = collector.run()
    print(result["corpus_df"].head())
