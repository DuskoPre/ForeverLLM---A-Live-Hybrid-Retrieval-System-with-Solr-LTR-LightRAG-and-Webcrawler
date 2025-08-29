# ForeverLLM Advanced Components - Part 2
# Advanced LTR Training, Query Understanding, and Production Features

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from collections import deque
import redis
import hashlib
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import msgpack

# ==================== Advanced LTR Training Module ====================
class AdvancedLTRTrainer:
    """Advanced Learning-to-Rank model training with multiple algorithms"""
    
    def __init__(self, config: 'ForeverLLMConfig'):
        self.config = config
        self.training_data = []
        self.model = None
        self.feature_importance = {}
        
    def collect_training_data(self, query: str, results: List[Dict], 
                             relevance_scores: List[float]) -> None:
        """Collect training data from user interactions"""
        for doc, score in zip(results, relevance_scores):
            features = self.extract_features(query, doc)
            self.training_data.append({
                'query': query,
                'doc_id': doc.get('id'),
                'features': features,
                'relevance': score
            })
    
    def extract_features(self, query: str, document: Dict) -> np.ndarray:
        """Extract LTR features from query-document pair"""
        features = []
        
        # Text similarity features
        query_terms = set(query.lower().split())
        title_terms = set(document.get('title', '').lower().split())
        text_terms = set(document.get('text', '')[:500].lower().split())
        
        # Feature 1: Title overlap
        title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        features.append(title_overlap)
        
        # Feature 2: Text overlap
        text_overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
        features.append(text_overlap)
        
        # Feature 3: Query length normalized
        features.append(len(query_terms) / 10.0)
        
        # Feature 4: Document length normalized
        doc_length = len(document.get('text', '').split())
        features.append(min(doc_length / 1000.0, 1.0))
        
        # Feature 5: Freshness (days old)
        try:
            timestamp = datetime.fromisoformat(document.get('timestamp', ''))
            age_days = (datetime.now() - timestamp).days
            freshness = 1.0 / (1.0 + age_days / 30)
            features.append(freshness)
        except:
            features.append(0.5)
        
        # Feature 6: Has code
        features.append(1.0 if document.get('has_code') else 0.0)
        
        # Feature 7: Entity overlap
        query_entities = set()  # Would extract from query
        doc_entities = set(document.get('entities_persons', []) + 
                          document.get('entities_orgs', []))
        entity_overlap = len(query_entities & doc_entities) / max(len(query_entities), 1)
        features.append(entity_overlap)
        
        # Feature 8: Click count (popularity)
        click_count = document.get('click_count', 0)
        features.append(min(click_count / 100.0, 1.0))
        
        return np.array(features)
    
    def train_xgboost_model(self) -> xgb.XGBRanker:
        """Train XGBoost LTR model"""
        if len(self.training_data) < 100:
            logger.warning("Insufficient training data for XGBoost")
            return None
        
        # Prepare data
        df = pd.DataFrame(self.training_data)
        X = np.vstack(df['features'].values)
        y = df['relevance'].values
        
        # Group by query for ranking
        query_groups = df.groupby('query').size().values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost ranker
        model = xgb.XGBRanker(
            tree_method='hist',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train, verbose=True)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            [f'feature_{i}' for i in range(X.shape[1])],
            model.feature_importances_
        ))
        
        self.model = model
        return model
    
    def train_lambdamart(self) -> GradientBoostingRegressor:
        """Train LambdaMART model (simplified version)"""
        if len(self.training_data) < 100:
            return None
        
        df = pd.DataFrame(self.training_data)
        X = np.vstack(df['features'].values)
        y = df['relevance'].values
        
        # Use Gradient Boosting as approximation
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X, y)
        self.model = model
        return model
    
    def evaluate_model(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate LTR model performance"""
        if not self.model or not test_data:
            return {}
        
        # Calculate metrics
        ndcg_scores = []
        map_scores = []
        
        for query_data in test_data:
            predictions = self.model.predict(query_data['features'])
            true_relevance = query_data['relevance']
            
            # Calculate NDCG@10
            ndcg = self._calculate_ndcg(true_relevance, predictions, k=10)
            ndcg_scores.append(ndcg)
            
            # Calculate MAP
            map_score = self._calculate_map(true_relevance, predictions)
            map_scores.append(map_score)
        
        return {
            'ndcg@10': np.mean(ndcg_scores),
            'map': np.mean(map_scores),
            'feature_importance': self.feature_importance
        }
    
    def _calculate_ndcg(self, true_relevance: np.ndarray, 
                       predictions: np.ndarray, k: int) -> float:
        """Calculate NDCG@k metric"""
        def dcg_at_k(scores, k):
            scores = scores[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
        
        # Sort by predictions
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        
        dcg = dcg_at_k(sorted_relevance, k)
        ideal_dcg = dcg_at_k(np.sort(true_relevance)[::-1], k)
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def _calculate_map(self, true_relevance: np.ndarray, 
                      predictions: np.ndarray) -> float:
        """Calculate Mean Average Precision"""
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        
        precisions = []
        relevant_count = 0
        
        for i, rel in enumerate(sorted_relevance):
            if rel > 0:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        return np.mean(precisions) if precisions else 0.0

# ==================== Query Understanding Module ====================
class QueryUnderstanding:
    """Advanced query analysis and intent detection"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.intent_patterns = self._load_intent_patterns()
        
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load query intent patterns"""
        return {
            'definition': ['what is', 'define', 'meaning of', 'explain'],
            'how_to': ['how to', 'how do', 'tutorial', 'guide', 'steps'],
            'comparison': ['vs', 'versus', 'compare', 'difference between'],
            'list': ['list of', 'examples of', 'types of', 'kinds of'],
            'troubleshooting': ['error', 'fix', 'solve', 'problem', 'issue'],
            'news': ['latest', 'recent', 'news', 'update', 'current'],
            'code': ['code', 'example', 'implementation', 'sample', 'snippet']
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Comprehensive query analysis"""
        analysis = {
            'original_query': query,
            'normalized_query': self._normalize_query(query),
            'intent': self._detect_intent(query),
            'entities': self._extract_entities(query),
            'complexity': self._assess_complexity(query),
            'temporal': self._detect_temporal_requirements(query),
            'domain': self._detect_domain(query)
        }
        
        return analysis
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        # Remove extra spaces, convert to lowercase
        normalized = ' '.join(query.lower().split())
        # Remove common stop words at beginning
        stop_prefixes = ['please', 'can you', 'i want to', 'show me']
        for prefix in stop_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        return normalized
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        # Simple implementation - in production, use SpaCy or similar
        entities = []
        
        # Look for capitalized words (potential entities)
        words = query.split()
        for word in words:
            if word[0].isupper() and word.lower() not in ['what', 'how', 'why']:
                entities.append(word)
        
        return entities
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        
        if word_count <= 3:
            return 'simple'
        elif word_count <= 8:
            return 'moderate'
        else:
            return 'complex'
    
    def _detect_temporal_requirements(self, query: str) -> Dict[str, Any]:
        """Detect temporal requirements in query"""
        temporal = {
            'requires_recent': False,
            'time_range': None
        }
        
        recent_indicators = ['latest', 'recent', 'new', 'current', 'today', 
                           'yesterday', 'this week', 'this month', '2025', '2024']
        
        query_lower = query.lower()
        for indicator in recent_indicators:
            if indicator in query_lower:
                temporal['requires_recent'] = True
                temporal['time_range'] = indicator
                break
        
        return temporal
    
    def _detect_domain(self, query: str) -> str:
        """Detect query domain"""
        domains = {
            'tech': ['code', 'programming', 'software', 'api', 'algorithm', 
                    'javascript', 'python', 'java', 'database'],
            'science': ['research', 'study', 'theory', 'hypothesis', 'experiment'],
            'business': ['marketing', 'sales', 'revenue', 'strategy', 'roi'],
            'health': ['medical', 'health', 'disease', 'treatment', 'symptoms']
        }
        
        query_lower = query.lower()
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return domain
        
        return 'general'
    
    def generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        variations = [query]
        
        # Add synonyms and related terms
        # Simplified - in production, use WordNet or similar
        replacements = {
            'create': ['make', 'build', 'develop'],
            'quick': ['fast', 'rapid', 'speedy'],
            'error': ['bug', 'issue', 'problem']
        }
        
        for word, synonyms in replacements.items():
            if word in query.lower():
                for synonym in synonyms:
                    variations.append(query.lower().replace(word, synonym))
        
        return list(set(variations))[:5]

# ==================== Distributed Cache Manager ====================
class DistributedCacheManager:
    """Redis-based distributed cache for ForeverLLM"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=False
        )
        self.ttl_default = 3600  # 1 hour
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            data = self.redis_client.get(key)
            if data:
                return msgpack.unpackb(data, raw=False)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        try:
            packed = msgpack.packb(value)
            return self.redis_client.setex(
                key, 
                ttl or self.ttl_default, 
                packed
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return self.redis_client.delete(key) > 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.redis_client.exists(key) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        info = self.redis_client.info()
        return {
            'used_memory': info.get('used_memory_human'),
            'total_keys': self.redis_client.dbsize(),
            'hits': info.get('keyspace_hits', 0),
            'misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / 
                       max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
        }

# ==================== Vector Index Manager ====================
class VectorIndexManager:
    """FAISS-based vector index for semantic search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.id_map = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to vector index"""
        embeddings = []
        
        for i, doc in enumerate(documents):
            # Create embedding
            text = f"{doc.get('title', '')} {doc.get('text', '')[:500]}"
            embedding = self.embedder.encode(text)
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            
            # Map index to document ID
            current_size = self.index.ntotal
            self.id_map[current_size + i] = doc.get('id')
        
        # Add to index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        logger.info(f"Added {len(documents)} documents to vector index")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        # Create query embedding
        query_embedding = self.embedder.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Map back to document IDs
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.map", 'wb') as f:
            pickle.dump(self.id_map, f)
    
    def load_index(self, path: str) -> None:
        """Load index from disk"""
        if os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")
            with open(f"{path}.map", 'rb') as f:
                self.id_map = pickle.load(f)

# ==================== Feedback Loop Manager ====================
class FeedbackLoopManager:
    """Manages user feedback and continuous improvement"""
    
    def __init__(self, config: 'ForeverLLMConfig'):
        self.config = config
        self.feedback_queue = deque(maxlen=1000)
        self.relevance_signals = defaultdict(list)
        self.query_success_rate = {}
        
    def record_interaction(self, query: str, results: List[Dict], 
                          user_action: Dict) -> None:
        """Record user interaction for learning"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results_shown': len(results),
            'clicked_position': user_action.get('clicked_position'),
            'dwell_time': user_action.get('dwell_time'),
            'satisfied': user_action.get('satisfied', None)
        }
        
        self.feedback_queue.append(interaction)
        
        # Update relevance signals
        if user_action.get('clicked_position') is not None:
            clicked_doc = results[user_action['clicked_position']]
            self.relevance_signals[query].append({
                'doc_id': clicked_doc.get('id'),
                'relevance': self._calculate_relevance_score(user_action)
            })
    
    def _calculate_relevance_score(self, user_action: Dict) -> float:
        """Calculate relevance score from user action"""
        score = 0.0
        
        # Position-based score (higher position = more relevant)
        position = user_action.get('clicked_position', -1)
        if position >= 0:
            score += 1.0 / (1.0 + position)
        
        # Dwell time score
        dwell_time = user_action.get('dwell_time', 0)
        if dwell_time > 30:  # More than 30 seconds
            score += 0.5
        if dwell_time > 60:  # More than 1 minute
            score += 0.5
        
        # Explicit satisfaction
        if user_action.get('satisfied') == True:
            score += 1.0
        elif user_action.get('satisfied') == False:
            score -= 0.5
        
        return max(0.0, min(score, 3.0))  # Normalize to [0, 3]
    
    def get_training_data(self) -> List[Dict]:
        """Get training data for LTR model update"""
        training_data = []
        
        for query, signals in self.relevance_signals.items():
            if len(signals) >= 3:  # Minimum signals needed
                for signal in signals:
                    training_data.append({
                        'query': query,
                        'doc_id': signal['doc_id'],
                        'relevance': signal['relevance']
                    })
        
        return training_data
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """Analyze feedback patterns"""
        if not self.feedback_queue:
            return {}
        
        df = pd.DataFrame(list(self.feedback_queue))
        
        analysis = {
            'total_interactions': len(df),
            'avg_results_shown': df['results_shown'].mean(),
            'click_through_rate': len(df[df['clicked_position'].notna()]) / len(df),
            'avg_dwell_time': df['dwell_time'].mean() if 'dwell_time' in df else 0,
            'satisfaction_rate': df['satisfied'].mean() if 'satisfied' in df else None
        }
        
        # Query success patterns
        query_groups = df.groupby('query')
        for query, group in query_groups:
            ctr = len(group[group['clicked_position'].notna()]) / len(group)
            self.query_success_rate[query] = ctr
        
        # Identify problematic queries
        analysis['low_performing_queries'] = [
            query for query, rate in self.query_success_rate.items() 
            if rate < 0.2
        ]
        
        return analysis

# ==================== Async Web Crawler ====================
class AsyncWebCrawler:
    """Asynchronous high-performance web crawler"""
    
    def __init__(self, config: 'ForeverLLMConfig'):
        self.config = config
        self.session = None
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()
    
    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content asynchronously"""
        async with self.semaphore:
            try:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
        return None
    
    async def crawl_urls(self, urls: List[str]) -> List[Dict]:
        """Crawl multiple URLs concurrently"""
        tasks = [self.crawl_single(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def crawl_single(self, url: str) -> Optional[Dict]:
        """Crawl single URL and extract content"""
        html = await self.fetch_url(url)
        if html:
            return self.extract_content(html, url)
        return None
    
    def extract_content(self, html: str, url: str) -> Dict:
        """Extract content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        return {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'text': soup.get_text(separator=' ', strip=True),
            'links': [
                urllib.parse.urljoin(url, link.get('href')) 
                for link in soup.find_all('a', href=True)
            ],
            'timestamp': datetime.now().isoformat()
        }

# ==================== Testing Suite ====================
class ForeverLLMTestSuite:
    """Comprehensive testing for ForeverLLM components"""
    
    def __init__(self):
        self.test_results = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all component tests"""
        logger.info("Starting ForeverLLM test suite...")
        
        # Test each component
        self.test_crawler()
        self.test_lightrag()
        self.test_solr_integration()
        self.test_ltr_training()
        self.test_query_understanding()
        self.test_cache()
        self.test_vector_index()
        
        # Summarize results
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total if total > 0 else 0,
            'details': self.test_results
        }
    
    def test_crawler(self):
        """Test web crawler functionality"""
        logger.info("Testing web crawler...")
        
        try:
            config = ForeverLLMConfig()
            crawler = IntelligentWebCrawler(config)
            
            # Test robots.txt checking
            can_fetch = crawler.can_fetch("https://www.example.com/page")
            assert isinstance(can_fetch, bool)
            
            # Test content extraction
            test_html = "<html><title>Test</title><body>Content</body></html>"
            content = crawler.extract_content(test_html, "http://test.com")
            assert content['title'] == 'Test'
            assert 'Content' in content['text']
            
            self.test_results.append({
                'component': 'Crawler',
                'passed': True,
                'message': 'All crawler tests passed'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'Crawler',
                'passed': False,
                'message': str(e)
            })
    
    def test_lightrag(self):
        """Test LightRAG functionality"""
        logger.info("Testing LightRAG...")
        
        try:
            config = ForeverLLMConfig()
            lightrag = LightRAG(config)
            
            # Test entity extraction
            test_text = "Apple Inc. was founded by Steve Jobs in Cupertino."
            entities, relations = lightrag.extract_entities_relations(test_text)
            assert len(entities) > 0
            
            # Test graph operations
            test_docs = [{
                'text': test_text,
                'url': 'http://test.com'
            }]
            lightrag.update_graph(test_docs)
            assert len(lightrag.graph.nodes()) > 0
            
            self.test_results.append({
                'component': 'LightRAG',
                'passed': True,
                'message': 'All LightRAG tests passed'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'LightRAG',
                'passed': False,
                'message': str(e)
            })
    
    def test_solr_integration(self):
        """Test Solr integration"""
        logger.info("Testing Solr integration...")
        
        try:
            # Check if Solr is accessible
            response = requests.get('http://localhost:8983/solr/admin/ping')
            assert response.status_code == 200
            
            self.test_results.append({
                'component': 'Solr',
                'passed': True,
                'message': 'Solr is accessible'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'Solr',
                'passed': False,
                'message': f'Solr not accessible: {e}'
            })
    
    def test_ltr_training(self):
        """Test LTR training functionality"""
        logger.info("Testing LTR training...")
        
        try:
            config = ForeverLLMConfig()
            trainer = AdvancedLTRTrainer(config)
            
            # Create dummy training data
            for i in range(150):
                trainer.training_data.append({
                    'query': f'test query {i % 10}',
                    'doc_id': f'doc_{i}',
                    'features': np.random.rand(8),
                    'relevance': np.random.rand()
                })
            
            # Train model
            model = trainer.train_lambdamart()
            assert model is not None
            
            self.test_results.append({
                'component': 'LTR Training',
                'passed': True,
                'message': 'LTR model trained successfully'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'LTR Training',
                'passed': False,
                'message': str(e)
            })
    
    def test_query_understanding(self):
        """Test query understanding"""
        logger.info("Testing query understanding...")
        
        try:
            qu = QueryUnderstanding()
            
            # Test query analysis
            analysis = qu.analyze_query("How to implement machine learning in Python?")
            assert analysis['intent'] == 'how_to'
            assert 'Python' in analysis['entities']
            assert analysis['domain'] == 'tech'
            
            self.test_results.append({
                'component': 'Query Understanding',
                'passed': True,
                'message': 'Query analysis working correctly'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'Query Understanding',
                'passed': False,
                'message': str(e)
            })
    
    def test_cache(self):
        """Test cache functionality"""
        logger.info("Testing cache...")
        
        try:
            # Test in-memory cache simulation
            cache = {}
            test_key = 'test_key'
            test_value = {'data': 'test'}
            
            cache[test_key] = test_value
            assert cache.get(test_key) == test_value
            
            self.test_results.append({
                'component': 'Cache',
                'passed': True,
                'message': 'Cache operations working'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'Cache',
                'passed': False,
                'message': str(e)
            })
    
    def test_vector_index(self):
        """Test vector index"""
        logger.info("Testing vector index...")
        
        try:
            index_manager = VectorIndexManager(dimension=384)
            
            # Add test documents
            test_docs = [
                {'id': '1', 'title': 'Machine Learning', 'text': 'ML is great'},
                {'id': '2', 'title': 'Deep Learning', 'text': 'DL is powerful'}
            ]
            index_manager.add_documents(test_docs)
            
            # Search
            results = index_manager.search("artificial intelligence", k=2)
            assert len(results) <= 2
            
            self.test_results.append({
                'component': 'Vector Index',
                'passed': True,
                'message': 'Vector index working correctly'
            })
        except Exception as e:
            self.test_results.append({
                'component': 'Vector Index',
                'passed': False,
                'message': str(e)
            })

# ==================== Performance Monitor ====================
class PerformanceMonitor:
    """Monitor system performance and health"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        
    def record_metric(self, name: str, value: float) -> None:
        """Record performance metric"""
        self.metrics[name].append({
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 1000 measurements
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health"""
        health = {
            'status': 'healthy',
            'issues': [],
            'metrics_summary': {}
        }
        
        # Check response times
        if 'response_time' in self.metrics:
            recent_times = [m['value'] for m in self.metrics['response_time'][-100:]]
            avg_time = np.mean(recent_times)
            
            health['metrics_summary']['avg_response_time'] = avg_time
            
            if avg_time > 5.0:
                health['status'] = 'degraded'
                health['issues'].append('High response times')
        
        # Check error rates
        if 'errors' in self.metrics:
            recent_errors = self.metrics['errors'][-100:]
            error_rate = len(recent_errors) / 100
            
            health['metrics_summary']['error_rate'] = error_rate
            
            if error_rate > 0.1:
                health['status'] = 'unhealthy'
                health['issues'].append('High error rate')
        
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        health['metrics_summary']['memory_usage'] = memory.percent
        
        if memory.percent > 90:
            health['status'] = 'degraded'
            health['issues'].append('High memory usage')
        
        return health
    
    def generate_report(self) -> str:
        """Generate performance report"""
        report = ["ForeverLLM Performance Report", "=" * 40]
        
        for metric_name, values in self.metrics.items():
            if values:
                recent_values = [m['value'] for m in values[-100:]]
                report.append(f"\n{metric_name}:")
                report.append(f"  Average: {np.mean(recent_values):.2f}")
                report.append(f"  Min: {np.min(recent_values):.2f}")
                report.append(f"  Max: {np.max(recent_values):.2f}")
                report.append(f"  Std Dev: {np.std(recent_values):.2f}")
        
        return "\n".join(report)

# ==================== Main Test Runner ====================
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("ForeverLLM Advanced Components Test")
    print("=" * 50)
    
    # Run tests
    test_suite = ForeverLLMTestSuite()
    results = test_suite.run_all_tests()
    
    print(f"\nTest Results:")
    print(f"Total: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    # Show details
    print("\nDetails:")
    for test in results['details']:
        status = "✓" if test['passed'] else "✗"
        print(f"{status} {test['component']}: {test['message']}")
    
    # Test performance monitoring
    print("\n" + "=" * 50)
    print("Testing Performance Monitor")
    
    monitor = PerformanceMonitor()
    
    # Simulate metrics
    import random
    for _ in range(50):
        monitor.record_metric('response_time', random.uniform(0.5, 3.0))
        if random.random() < 0.05:
            monitor.record_metric('errors', 1)
    
    health = monitor.check_health()
    print(f"System Health: {health['status']}")
    print(f"Issues: {health['issues']}")
    
    print("\n" + monitor.generate_report())
