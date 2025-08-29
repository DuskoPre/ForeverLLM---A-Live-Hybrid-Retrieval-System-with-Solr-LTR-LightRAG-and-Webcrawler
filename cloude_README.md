# ForeverLLM - Live Hybrid Retrieval System

A continuous learning LLM knowledge base system combining Solr LTR, LightRAG, and intelligent web crawling.

## Features

- **Apache Solr with Learning-to-Rank (LTR)**: Advanced relevance ranking beyond traditional search
- **LightRAG Integration**: Graph-based dual-level retrieval for comprehensive context
- **Intelligent Web Crawler**: Continuous knowledge acquisition with NLP enrichment
- **Continuous Learning**: Adapts to user queries and improves over time
- **Hybrid Search**: Combines keyword, semantic, and graph-based retrieval

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- 8GB+ RAM recommended

### Installation

1. Run the setup script:
```bash
chmod +x setup_forever_llm.sh
./setup_forever_llm.sh
```

2. Start the ForeverLLM system:
```bash
python forever_llm.py
```

### Usage

```python
from forever_llm import ForeverLLM, ForeverLLMConfig

# Initialize
config = ForeverLLMConfig()
forever_llm = ForeverLLM(config)

# Process query
result = forever_llm.process_query("What is Learning to Rank?")
print(result['context'])
```

## Architecture

### Components

1. **Web Crawler Module**
   - Respects robots.txt
   - NLP entity extraction
   - Incremental crawling
   - Domain-specific targeting

2. **LightRAG Module**
   - Knowledge graph construction
   - Dual-level retrieval
   - Entity and relation extraction
   - Graph-based context enrichment

3. **Solr LTR Module**
   - Hybrid search (keyword + vector)
   - Feature-based ranking
   - Online learning capability
   - Relevance feedback integration

4. **ForeverLLM Core**
   - Query processing pipeline
   - Result merging and ranking
   - Continuous improvement loop
   - Cache management

## Configuration

Edit `ForeverLLMConfig` in the main script:

```python
config = ForeverLLMConfig(
    solr_url="http://localhost:8983/solr/forever_llm",
    crawl_depth=3,
    max_pages_per_domain=100,
    ltr_retrain_interval=100
)
```

## Performance

- Query latency: <1s for cached, 2-5s for new queries
- Indexing speed: ~100 docs/second
- Graph update: Incremental, <100ms per document
- Memory usage: ~2GB base + data

## Development

### Adding New Features

1. Extend LTR features in `ltr_features.json`
2. Add new entity types in LightRAG
3. Implement custom crawlers for specific domains

### Training Custom Models

The system supports custom LTR model training:

```python
# Collect training data
training_data = forever_llm.collect_relevance_feedback()

# Train new model
forever_llm.train_ltr_model(training_data)
```

## Monitoring

Check system statistics:

```python
stats = forever_llm.get_system_stats()
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit PRs with tests.

## Support

For issues and questions, please use GitHub Issues.
