#!/usr/bin/env python3
"""
ForeverLLM Example Usage Script
"""

import time
from forever_llm import ForeverLLM, ForeverLLMConfig

def main():
    print("ForeverLLM Example Usage")
    print("=" * 50)
    
    # Initialize with custom configuration
    config = ForeverLLMConfig(
        solr_url="http://localhost:8983/solr/forever_llm",
        crawl_depth=2,
        max_pages_per_domain=50
    )
    
    # Create ForeverLLM instance
    forever_llm = ForeverLLM(config)
    
    # Example 1: Simple query
    print("\n1. Simple Query Example:")
    result = forever_llm.process_query("What is machine learning?")
    print(f"   Found {len(result['context']['documents'])} documents")
    print(f"   Processing time: {result['processing_time']:.2f}s")
    
    # Example 2: Technical query
    print("\n2. Technical Query Example:")
    result = forever_llm.process_query("How to implement Learning to Rank in Solr?")
    print(f"   Found {len(result['context']['documents'])} documents")
    if result['context']['documents']:
        print(f"   Top result: {result['context']['documents'][0]['title']}")
    
    # Example 3: Entity-focused query
    print("\n3. Entity-focused Query Example:")
    result = forever_llm.process_query("OpenAI ChatGPT applications")
    print(f"   Identified entities: {len(result['context']['entities'])}")
    for entity in result['context']['entities'][:3]:
        print(f"   - {entity['entity']}")
    
    # Example 4: Trigger knowledge expansion
    print("\n4. Knowledge Expansion Example:")
    forever_llm.schedule_topic_crawl("quantum computing")
    time.sleep(5)  # Wait for crawling
    
    # Check system stats
    print("\n5. System Statistics:")
    stats = forever_llm.get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Example 6: Continuous learning simulation
    print("\n6. Simulating Continuous Learning:")
    queries = [
        "Python programming best practices",
        "JavaScript async await",
        "Docker container orchestration",
        "Kubernetes deployment strategies",
        "React hooks tutorial"
    ]
    
    for query in queries:
        result = forever_llm.process_query(query)
        print(f"   Processed: {query[:40]}...")
        time.sleep(1)
    
    # Final stats
    print("\n7. Final Statistics:")
    stats = forever_llm.get_system_stats()
    print(f"   Total queries processed: {stats['queries_processed']}")
    print(f"   Knowledge graph size: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")
    print(f"   Document collection: {stats['total_documents']} documents")

if __name__ == "__main__":
    main()
