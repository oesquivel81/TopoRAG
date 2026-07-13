# Colab export para TopoRAG

Esta carpeta contiene una clase reutilizable para construir el corpus arXiv, limpiarlo en un flujo condensado y dejarlo listo para usarse en la siguiente etapa del pipeline.

## Uso en Colab

```python
!pip install -q -r /content/TopoRAG/colab_export/requirements.txt

from pathlib import Path
import sys
sys.path.append('/content/TopoRAG/colab_export')

from arxiv_corpus_collector import ArxivCorpusCollector

collector = ArxivCorpusCollector(
    search_queries={
        "differential_geometry": 'cat:math.DG AND (all:"differential geometry" OR all:"Riemannian geometry")',
        "algebraic_topology": 'cat:math.AT AND (all:"algebraic topology" OR all:"homology" OR all:"cohomology")',
        "topological_data_analysis": '(cat:math.AT OR cat:cs.LG OR cat:stat.ML) AND (all:"persistent homology" OR all:"topological data analysis")',
    },
    project_root='/content/TopoRAG',
    max_results_per_query=10,
)

result = collector.run()
corpus_df = result['corpus_df']
display(corpus_df.head())
```

El resultado final queda disponible en:
- `result['corpus_df']` para el corpus limpio y consolidado
- `result['manifest_df']` para el manifiesto
- `result['pages_df']` para las páginas crudas
- `result['cleaning_quality_df']` para el reporte de calidad
- `result['cleaning_summary']` para el resumen del proceso

Además se exportan automáticamente:
- `data/processed/arxiv_pages_clean.parquet`
- `data/processed/arxiv_pages_clean.jsonl`
- `data/processed/page_quality_report.csv`
