# ForeverLLM Production Deployment - Part 3
# REST API, Monitoring Dashboard, and Production Optimizations

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import plotly.graph_objs as go
import plotly.utils
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from concurrent.futures import ThreadPoolExecutor
import schedule
import threading
import requests
import logging
import os
import sys
import psutil
from pathlib import Path
import uuid
from dataclasses import dataclass
from abc import ABC, abstractmethod


from Advanced_Components import FeedbackLoopManager
from Advanced_Components import AsyncWebCrawler
from Advanced_Components import VectorIndexManager
from Advanced_Components import AdvancedLTRTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
@dataclass
class ForeverLLMConfig:
    """Configuration for ForeverLLM system"""
    solr_url: str = "http://localhost:8983/solr/forever_llm"
    redis_host: str = "localhost"
    redis_port: int = 6379
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_workers: int = 4
    cache_ttl: int = 3600
    crawl_delay: float = 1.0
    max_crawl_depth: int = 3

# ==================== Mock Classes for Dependencies ====================
class MockSolrManager:
    """Mock Solr manager for demonstration"""
    def __init__(self, config):
        self.config = config
        
    def index_documents(self, documents):
        logger.info(f"Indexed {len(documents)} documents")
        return True
    
    def search(self, query, filters=None, limit=10):
        return {
            'documents': [
                {'id': f'doc_{i}', 'title': f'Result {i}', 'snippet': f'Content for {query}'}
                for i in range(min(limit, 5))
            ]
        }

class MockLightRAG:
    """Mock LightRAG for demonstration"""
    def __init__(self, config):
        self.config = config
        self.nodes = 1000
        self.edges = 2500
        
    def update_graph(self, documents):
        logger.info(f"Updated graph with {len(documents)} documents")
        
    def get_related_entities(self, query):
        return [{'entity': 'example', 'relevance': 0.8}]
    
    def _optimize_knowledge_graph(self):
        logger.info("Knowledge graph optimized")


# Mock the main ForeverLLM class
class ForeverLLM:
    """Mock ForeverLLM main class"""
    def __init__(self, config):
        self.config = config
        self.solr_manager = MockSolrManager(config)
        self.lightrag = MockLightRAG(config)
        self.crawler = MockCrawler(config)
        self.feedback_manager = MockFeedbackManager(config)
        self.query_log = []
        self.crawler_queue = []
        
    def process_query(self, query, use_cache=True):
        start_time = time.time()
        
        # Mock query processing
        results = self.solr_manager.search(query)
        entities = self.lightrag.get_related_entities(query)
        
        processing_time = time.time() - start_time
        
        result = {
            'context': {
                'documents': results['documents'],
                'entities': entities,
                'relations': []
            },
            'processing_time': processing_time,
            'cached': use_cache and len(self.query_log) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.query_log.append({
            'query': query,
            'timestamp': datetime.now(),
            'processing_time': processing_time
        })
        
        return result
    
    def get_system_stats(self):
        return {
            'total_documents': 10000,
            'graph_nodes': self.lightrag.nodes,
            'graph_edges': self.lightrag.edges,
            'queries_processed': len(self.query_log),
            'cache_hits': max(0, len(self.query_log) - 10),
            'crawl_queue_size': len(self.crawler_queue)
        }

# ==================== API Models ====================
class QueryRequest(BaseModel):
    """API request model for queries"""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[Dict[str, Any]] = None
    max_results: int = Field(default=10, ge=1, le=100)
    use_cache: bool = True
    include_graph: bool = True
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """API response model for queries"""
    query: str
    results: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    processing_time: float
    cached: bool
    session_id: str
    timestamp: str

class FeedbackRequest(BaseModel):
    """API request model for feedback"""
    session_id: str
    query_id: str
    doc_id: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0, le=5)
    clicked_position: Optional[int] = None
    dwell_time: Optional[int] = None
    satisfied: Optional[bool] = None

class CrawlRequest(BaseModel):
    """API request model for triggering crawl"""
    urls: List[str] = Field(..., min_items=1, max_items=100)
    depth: int = Field(default=2, ge=1, le=5)
    priority: str = Field(default="normal", pattern="^(low|normal|high)$")

# ==================== Metrics for Monitoring ====================
# Prometheus metrics
query_counter = Counter('foreverllm_queries_total', 'Total number of queries processed')
query_histogram = Histogram('foreverllm_query_duration_seconds', 'Query processing duration')
cache_hits = Counter('foreverllm_cache_hits_total', 'Total cache hits')
cache_misses = Counter('foreverllm_cache_misses_total', 'Total cache misses')
active_sessions = Gauge('foreverllm_active_sessions', 'Number of active sessions')
documents_indexed = Counter('foreverllm_documents_indexed_total', 'Total documents indexed')
graph_nodes_gauge = Gauge('foreverllm_graph_nodes', 'Number of nodes in knowledge graph')
error_counter = Counter('foreverllm_errors_total', 'Total errors', ['error_type'])

# ==================== FastAPI Application ====================
class ForeverLLMAPI:
    """Production API for ForeverLLM system"""
    
    def __init__(self, config: ForeverLLMConfig = None):
        self.app = FastAPI(
            title="ForeverLLM API",
            description="Continuous Learning LLM Knowledge Base System",
            version="1.0.0"
        )
        
        self.config = config or ForeverLLMConfig()
        self.forever_llm = None  # Will be initialized in startup
        self.sessions = {}
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = "your-secret-key-change-in-production"
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks executor
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize system on startup"""
            logger.info("Starting ForeverLLM API...")
            self.forever_llm = ForeverLLM(self.config)
            
            # Start background tasks
            self._start_background_tasks()
            
            logger.info("ForeverLLM API started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            logger.info("Shutting down ForeverLLM API...")
            self.executor.shutdown(wait=True)
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve monitoring dashboard"""
            return self._generate_dashboard_html()
        
        @self.app.post("/api/v1/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
            """Process a search query"""
            query_counter.inc()
            
            with query_histogram.time():
                try:
                    # Process query
                    result = self.forever_llm.process_query(
                        query=request.query,
                        use_cache=request.use_cache
                    )
                    
                    # Track cache usage
                    if result.get('cached'):
                        cache_hits.inc()
                    else:
                        cache_misses.inc()
                    
                    # Create session if needed
                    session_id = request.session_id or self._create_session()
                    self.sessions[session_id] = {
                        'last_query': request.query,
                        'timestamp': datetime.now()
                    }
                    
                    # Format response
                    response = QueryResponse(
                        query=request.query,
                        results=result['context']['documents'][:request.max_results],
                        entities=result['context'].get('entities', []) if request.include_graph else [],
                        relations=result['context'].get('relations', []) if request.include_graph else [],
                        processing_time=result['processing_time'],
                        cached=result.get('cached', False),
                        session_id=session_id,
                        timestamp=result['timestamp']
                    )
                    
                    # Schedule background analysis
                    background_tasks.add_task(
                        self._analyze_query_pattern,
                        request.query,
                        result
                    )
                    
                    return response
                    
                except Exception as e:
                    error_counter.labels(error_type='query_processing').inc()
                    logger.error(f"Query processing error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/feedback")
        async def submit_feedback(request: FeedbackRequest):
            """Submit relevance feedback"""
            try:
                # Record feedback
                self.forever_llm.feedback_manager.record_interaction(
                    query=self.sessions.get(request.session_id, {}).get('last_query', ''),
                    results=[],  # Would need to store results in session
                    user_action={
                        'clicked_position': request.clicked_position,
                        'dwell_time': request.dwell_time,
                        'satisfied': request.satisfied
                    }
                )
                
                return {"status": "success", "message": "Feedback recorded"}
                
            except Exception as e:
                error_counter.labels(error_type='feedback').inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/crawl")
        async def trigger_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
            """Trigger web crawling"""
            try:
                # Schedule crawl based on priority
                if request.priority == "high":
                    # Process immediately
                    background_tasks.add_task(
                        self._execute_crawl,
                        request.urls,
                        request.depth
                    )
                    message = "High priority crawl started"
                else:
                    # Add to queue
                    self.forever_llm.crawler_queue.extend(request.urls)
                    message = f"Added {len(request.urls)} URLs to crawl queue"
                
                return {"status": "success", "message": message}
                
            except Exception as e:
                error_counter.labels(error_type='crawl').inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/stats")
        async def get_statistics():
            """Get system statistics"""
            try:
                stats = self.forever_llm.get_system_stats()
                
                # Update Prometheus metrics
                graph_nodes_gauge.set(stats['graph_nodes'])
                active_sessions.set(len(self.sessions))
                
                # Add API-specific stats
                stats['active_sessions'] = len(self.sessions)
                stats['api_queries_total'] = query_counter._value.get()
                
                return stats
                
            except Exception as e:
                error_counter.labels(error_type='stats').inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check Solr
                solr_healthy = self._check_solr_health()
                
                # Check memory
                memory = psutil.virtual_memory()
                
                health = {
                    'status': 'healthy' if solr_healthy and memory.percent < 90 else 'degraded',
                    'components': {
                        'solr': 'healthy' if solr_healthy else 'unhealthy',
                        'memory_usage': f"{memory.percent:.1f}%",
                        'api': 'healthy'
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                return health
                
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.get("/api/v1/suggestions")
        async def get_suggestions(q: str):
            """Get query suggestions"""
            try:
                # Generate suggestions based on query log
                suggestions = self._generate_suggestions(q)
                return {"suggestions": suggestions}
                
            except Exception as e:
                error_counter.labels(error_type='suggestions').inc()
                raise HTTPException(status_code=500, detail=str(e))
    
    def _create_session(self) -> str:
        """Create new session ID"""
        return str(uuid.uuid4())
    
    def _check_solr_health(self) -> bool:
        """Check if Solr is healthy"""
        try:
            response = requests.get(f"{self.config.solr_url}/admin/ping")
            return response.status_code == 200
        except:
            return False
    
    async def _analyze_query_pattern(self, query: str, result: Dict):
        """Analyze query pattern in background"""
        # This would implement pattern analysis for continuous learning
        pass
    
    async def _execute_crawl(self, urls: List[str], depth: int):
        """Execute web crawl in background"""
        try:
            results = self.forever_llm.crawler.targeted_crawl(urls)
            
            if results:
                # Index documents
                self.forever_llm.solr_manager.index_documents(results)
                
                # Update graph
                self.forever_llm.lightrag.update_graph(results)
                
                documents_indexed.inc(len(results))
                logger.info(f"Crawled and indexed {len(results)} documents")
                
        except Exception as e:
            error_counter.labels(error_type='background_crawl').inc()
            logger.error(f"Background crawl error: {e}")
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """Generate query suggestions"""
        suggestions = []
        
        # Get recent successful queries
        recent_queries = [
            log['query'] for log in self.forever_llm.query_log[-100:]
            if query.lower() in log['query'].lower()
        ]
        
        # Deduplicate and limit
        seen = set()
        for q in recent_queries:
            if q not in seen and q != query:
                suggestions.append(q)
                seen.add(q)
                if len(suggestions) >= 5:
                    break
        
        return suggestions
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        # Schedule periodic tasks
        schedule.every(30).minutes.do(self._cleanup_sessions)
        schedule.every(1).hours.do(self._optimize_indices)
        schedule.every(6).hours.do(self._retrain_models)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _cleanup_sessions(self):
        """Clean up old sessions"""
        cutoff = datetime.now() - timedelta(hours=24)
        old_sessions = [
            sid for sid, data in self.sessions.items()
            if data['timestamp'] < cutoff
        ]
        
        for sid in old_sessions:
            del self.sessions[sid]
        
        logger.info(f"Cleaned up {len(old_sessions)} old sessions")
    
    def _optimize_indices(self):
        """Optimize search indices"""
        try:
            # Optimize Solr index (mock)
            logger.info("Solr index optimized")
            
            # Optimize knowledge graph
            self.forever_llm.lightrag._optimize_knowledge_graph()
            
            logger.info("Indices optimized")
        except Exception as e:
            logger.error(f"Index optimization error: {e}")
    
    def _retrain_models(self):
        """Retrain ranking models"""
        try:
            # Get training data from feedback
            training_data = self.forever_llm.feedback_manager.get_training_data()
            
            if len(training_data) >= 100:
                # Train new LTR model
                trainer = MockAdvancedLTRTrainer(self.config)
                trainer.training_data = training_data
                model = trainer.train_xgboost_model()
                
                if model:
                    # Deploy new model
                    logger.info("Retrained and deployed new LTR model")
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    def _generate_dashboard_html(self) -> str:
        """Generate monitoring dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ForeverLLM Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                }
                h1 {
                    color: white;
                    text-align: center;
                    font-size: 2.5em;
                    margin-bottom: 30px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .card {
                    background: white;
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    transition: transform 0.3s;
                }
                .card:hover {
                    transform: translateY(-5px);
                }
                .stat {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }
                .stat-label {
                    color: #6c757d;
                    font-size: 0.9em;
                }
                .stat-value {
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #495057;
                }
                .chart-container {
                    height: 300px;
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-healthy { background: #28a745; }
                .status-degraded { background: #ffc107; }
                .status-unhealthy { background: #dc3545; }
                .query-box {
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }
                .query-input {
                    width: 100%;
                    padding: 15px;
                    font-size: 1.1em;
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                    outline: none;
                    transition: border-color 0.3s;
                }
                .query-input:focus {
                    border-color: #667eea;
                }
                .query-button {
                    margin-top: 15px;
                    padding: 12px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 1em;
                    cursor: pointer;
                    transition: transform 0.2s;
                }
                .query-button:hover {
                    transform: scale(1.05);
                }
                .results {
                    margin-top: 20px;
                    max-height: 400px;
                    overflow-y: auto;
                }
                .result-item {
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }
                .result-title {
                    font-weight: bold;
                    color: #495057;
                    margin-bottom: 5px;
                }
                .result-snippet {
                    color: #6c757d;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 ForeverLLM Dashboard</h1>
                
                <!-- Query Interface -->
                <div class="query-box">
                    <h2>Test Query</h2>
                    <input type="text" 
                           class="query-input" 
                           id="queryInput" 
                           placeholder="Enter your query..."
                           onkeypress="if(event.key==='Enter') executeQuery()">
                    <button class="query-button" onclick="executeQuery()">Search</button>
                    <div id="results" class="results"></div>
                </div>
                
                <!-- Statistics Grid -->
                <div class="grid">
                    <!-- System Status -->
                    <div class="card">
                        <h3>System Status</h3>
                        <div class="stat">
                            <span class="stat-label">
                                <span class="status-indicator status-healthy"></span>
                                API Status
                            </span>
                            <span class="stat-value">Healthy</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Uptime</span>
                            <span class="stat-value" id="uptime">--</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Memory Usage</span>
                            <span class="stat-value" id="memory">--</span>
                        </div>
                    </div>
                    
                    <!-- Query Statistics -->
                    <div class="card">
                        <h3>Query Statistics</h3>
                        <div class="stat">
                            <span class="stat-label">Total Queries</span>
                            <span class="stat-value" id="totalQueries">--</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Cache Hit Rate</span>
                            <span class="stat-value" id="cacheHitRate">--</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Avg Response Time</span>
                            <span class="stat-value" id="avgResponseTime">--</span>
                        </div>
                    </div>
                    
                    <!-- Knowledge Base Stats -->
                    <div class="card">
                        <h3>Knowledge Base</h3>
                        <div class="stat">
                            <span class="stat-label">Documents</span>
                            <span class="stat-value" id="totalDocs">--</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Graph Nodes</span>
                            <span class="stat-value" id="graphNodes">--</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Graph Edges</span>
                            <span class="stat-value" id="graphEdges">--</span>
                        </div>
                    </div>
                </div>
                
                <!-- Charts -->
                <div class="grid">
                    <div class="card">
                        <h3>Query Volume (Last 24h)</h3>
                        <div id="queryVolumeChart" class="chart-container"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Response Times</h3>
                        <div id="responseTimeChart" class="chart-container"></div>
                    </div>
                </div>
                
                <!-- Recent Queries -->
                <div class="card">
                    <h3>Recent Queries</h3>
                    <div id="recentQueries"></div>
                </div>
            </div>
            
            <script>
                let startTime = Date.now();
                
                // Update statistics
                async function updateStats() {
                    try {
                        const response = await fetch('/api/v1/stats');
                        const stats = await response.json();
                        
                        document.getElementById('totalQueries').textContent = stats.queries_processed || 0;
                        document.getElementById('totalDocs').textContent = stats.total_documents || 0;
                        document.getElementById('graphNodes').textContent = stats.graph_nodes || 0;
                        document.getElementById('graphEdges').textContent = stats.graph_edges || 0;
                        
                        // Calculate cache hit rate
                        const cacheHits = stats.cache_hits || 0;
                        const totalQueries = stats.queries_processed || 1;
                        const hitRate = ((cacheHits / totalQueries) * 100).toFixed(1);
                        document.getElementById('cacheHitRate').textContent = hitRate + '%';
                        
                        // Update uptime
                        const uptime = Math.floor((Date.now() - startTime) / 1000);
                        const hours = Math.floor(uptime / 3600);
                        const minutes = Math.floor((uptime % 3600) / 60);
                        document.getElementById('uptime').textContent = `${hours}h ${minutes}m`;
                        
                    } catch (error) {
                        console.error('Error fetching stats:', error);
                    }
                }
                
                // Execute query
                async function executeQuery() {
                    const queryInput = document.getElementById('queryInput');
                    const query = queryInput.value.trim();
                    
                    if (!query) return;
                    
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = '<div class="result-item">Searching...</div>';
                    
                    try {
                        const response = await fetch('/api/v1/query', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                query: query,
                                max_results: 5,
                                include_graph: true
                            })
                        });
                        
                        const data = await response.json();
                        
                        // Display results
                        resultsDiv.innerHTML = '';
                        
                        if (data.results && data.results.length > 0) {
                            data.results.forEach(result => {
                                const item = document.createElement('div');
                                item.className = 'result-item';
                                item.innerHTML = `
                                    <div class="result-title">${result.title || 'Untitled'}</div>
                                    <div class="result-snippet">${result.snippet || ''}</div>
                                `;
                                resultsDiv.appendChild(item);
                            });
                        } else {
                            resultsDiv.innerHTML = '<div class="result-item">No results found</div>';
                        }
                        
                        // Update stats
                        updateStats();
                        
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="result-item">Error: ${error.message}</div>`;
                    }
                }
                
                // Initialize charts
                function initCharts() {
                    // Query volume chart
                    const hours = Array.from({length: 24}, (_, i) => i);
                    const queryVolume = hours.map(() => Math.floor(Math.random() * 100));
                    
                    Plotly.newPlot('queryVolumeChart', [{
                        x: hours,
                        y: queryVolume,
                        type: 'bar',
                        marker: {color: '#667eea'}
                    }], {
                        margin: {t: 20, r: 20, b: 40, l: 40},
                        xaxis: {title: 'Hour'},
                        yaxis: {title: 'Queries'}
                    });
                    
                    // Response time chart
                    const times = Array.from({length: 50}, () => Math.random() * 3 + 0.5);
                    
                    Plotly.newPlot('responseTimeChart', [{
                        y: times,
                        type: 'box',
                        marker: {color: '#764ba2'}
                    }], {
                        margin: {t: 20, r: 20, b: 40, l: 40},
                        yaxis: {title: 'Response Time (s)'}
                    });
                }
                
                // Initialize
                updateStats();
                initCharts();
                setInterval(updateStats, 5000);
            </script>
        </body>
        </html>
        """

# ==================== Kubernetes Deployment ====================
def generate_kubernetes_deployment():
    """Generate Kubernetes deployment configuration"""
    
    deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: foreverllm
  labels:
    app: foreverllm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: foreverllm
  template:
    metadata:
      labels:
        app: foreverllm
    spec:
      containers:
      - name: foreverllm-api
        image: foreverllm:latest
        ports:
        - containerPort: 8000
        env:
        - name: SOLR_URL
          value: "http://solr-service:8983/solr/forever_llm"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: foreverllm-service
spec:
  selector:
    app: foreverllm
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: foreverllm-crawler
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: crawler
            image: foreverllm:latest
            command:
            - python
            - -c
            - "from forever_llm import ForeverLLM; llm = ForeverLLM(); llm.schedule_topic_crawl('trending')"
          restartPolicy: OnFailure
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: foreverllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: foreverllm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
    
    with open('k8s-deployment.yaml', 'w') as f:
        f.write(deployment)
    
    logger.info("Kubernetes deployment configuration written to k8s-deployment.yaml")

# ==================== Docker Configuration ====================
def generate_dockerfile():
    """Generate Dockerfile for containerization"""
    
    dockerfile = """
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "Production_Deployment:create_app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    # Generate requirements.txt
    requirements = """
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
prometheus-client==0.19.0
plotly==5.17.0
passlib[bcrypt]==1.7.4
PyJWT==2.8.0
requests==2.31.0
schedule==1.2.0
psutil==5.9.6
redis==5.0.1
pysolr==3.9.0
aiofiles==23.2.1
python-multipart==0.0.6
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    logger.info("Dockerfile and requirements.txt generated")

# ==================== Docker Compose ====================
def generate_docker_compose():
    """Generate Docker Compose configuration"""
    
    compose = """
version: '3.8'

services:
  foreverllm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SOLR_URL=http://solr:8983/solr/forever_llm
      - REDIS_HOST=redis
    depends_on:
      - solr
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  solr:
    image: solr:9.4
    ports:
      - "8983:8983"
    volumes:
      - solr_data:/var/solr
      - ./solr-config:/opt/solr-8.11.2/server/solr/configsets/forever_llm
    command:
      - solr-precreate
      - forever_llm
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  solr_data:
  redis_data:
  prometheus_data:
  grafana_data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose)
    
    # Generate Prometheus config
    prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'foreverllm-api'
    static_configs:
      - targets: ['foreverllm-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
"""
    
    with open('prometheus.yml', 'w') as f:
        f.write(prometheus_config)
    
    logger.info("Docker Compose configuration generated")

# ==================== Production Utilities ====================
class ProductionManager:
    """Manage production deployment tasks"""
    
    def __init__(self, config: ForeverLLMConfig):
        self.config = config
    
    def setup_logging(self):
        """Setup production logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/foreverllm.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'data', 'models', 'cache']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        logger.info("Created necessary directories")
    
    def validate_config(self):
        """Validate production configuration"""
        required_vars = ['SOLR_URL', 'REDIS_HOST']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        return True

# ==================== Application Factory ====================
def create_app(config: ForeverLLMConfig = None) -> FastAPI:
    """Create and configure FastAPI application"""
    if config is None:
        config = ForeverLLMConfig()
        
        # Override with environment variables
        config.solr_url = os.getenv('SOLR_URL', config.solr_url)
        config.redis_host = os.getenv('REDIS_HOST', config.redis_host)
        config.api_host = os.getenv('API_HOST', config.api_host)
        config.api_port = int(os.getenv('API_PORT', config.api_port))
    
    # Setup production environment
    manager = ProductionManager(config)
    manager.setup_logging()
    manager.create_directories()
    
    if not manager.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Create API instance
    api = ForeverLLMAPI(config)
    
    return api.app

# ==================== Main Entry Point ====================
def main():
    """Main entry point for production deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ForeverLLM Production Deployment')
    parser.add_argument('--generate-configs', action='store_true', 
                       help='Generate deployment configurations')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    if args.generate_configs:
        logger.info("Generating deployment configurations...")
        generate_kubernetes_deployment()
        generate_dockerfile()
        generate_docker_compose()
        logger.info("Deployment configurations generated successfully!")
        return
    
    # Create configuration
    config = ForeverLLMConfig()
    config.api_host = args.host
    config.api_port = args.port
    
    # Create application
    app = create_app(config)
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        access_log=True,
        loop='auto'
    )

if __name__ == "__main__":
    main()
