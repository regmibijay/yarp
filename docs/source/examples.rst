Examples
========

This section provides practical examples of using YARP for different scenarios.

Document Search System
-----------------------

Building a simple document search system:

.. code-block:: python

    from yarp.vector_index import LocalMemoryIndex
    import json

    # Load documents from a JSON file
    def load_documents(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return [doc['content'] for doc in data['documents']]

    # Create search system
    class DocumentSearcher:
        def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
            self.index = LocalMemoryIndex(documents, model_name)
            self.index.process()
        
        def search(self, query, max_results=5):
            results = self.index.query(query, top_k=max_results)
            return [(r.document, r.matching_score) for r in results]
        
        def add_document(self, document):
            self.index.add(document)
        
        def save(self, path):
            self.index.backup(path)
        
        @classmethod
        def load(cls, path):
            instance = cls.__new__(cls)
            instance.index = LocalMemoryIndex.load(path)
            return instance

    # Usage
    documents = load_documents('my_documents.json')
    searcher = DocumentSearcher(documents)
    
    results = searcher.search("machine learning algorithms")
    for doc, score in results:
        print(f"[{score:.1f}] {doc[:100]}...")

FAQ System
----------

Creating an intelligent FAQ system:

.. code-block:: python

    from yarp.vector_index import LocalMemoryIndex

    class FAQSystem:
        def __init__(self):
            # FAQ data: questions and their answers
            self.faqs = {
                "How do I install the software?": "Run 'pip install our-software' to install.",
                "What are the system requirements?": "Python 3.8+, 4GB RAM minimum.",
                "How do I reset my password?": "Click 'Forgot Password' on the login page.",
                "Where can I find the documentation?": "Visit our website's docs section.",
                "How do I contact support?": "Email us at support@example.com",
            }
            
            # Create index from questions
            questions = list(self.faqs.keys())
            self.index = LocalMemoryIndex(questions)
            self.index.process()
        
        def ask(self, user_question, threshold=30.0):
            results = self.index.query(user_question, top_k=3)
            
            best_matches = []
            for result in results:
                if result.matching_score >= threshold:
                    question = result.document
                    answer = self.faqs[question]
                    confidence = result.matching_score
                    best_matches.append((question, answer, confidence))
            
            return best_matches
        
        def add_faq(self, question, answer):
            self.faqs[question] = answer
            self.index.add(question)

    # Usage
    faq = FAQSystem()
    
    user_query = "how to install this program?"
    matches = faq.ask(user_query)
    
    if matches:
        for question, answer, confidence in matches:
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"Confidence: {confidence:.1f}%\n")
    else:
        print("No matching FAQ found. Please contact support.")

Content Recommendation
----------------------

Building a content recommendation system:

.. code-block:: python

    from yarp.vector_index import LocalMemoryIndex
    from typing import List, Dict

    class ContentRecommender:
        def __init__(self, content_items: List[Dict]):
            """
            content_items: List of dicts with 'id', 'title', 'description', 'tags'
            """
            self.items = {item['id']: item for item in content_items}
            
            # Create searchable text from title, description, and tags
            self.searchable_texts = []
            self.item_ids = []
            
            for item in content_items:
                search_text = f"{item['title']} {item['description']} {' '.join(item['tags'])}"
                self.searchable_texts.append(search_text)
                self.item_ids.append(item['id'])
            
            # Build index
            self.index = LocalMemoryIndex(self.searchable_texts)
            self.index.process()
        
        def recommend_by_interest(self, interests: str, count: int = 5):
            """Recommend content based on user interests"""
            results = self.index.query(interests, top_k=count)
            
            recommendations = []
            for result in results:
                # Find the corresponding item
                text_index = self.searchable_texts.index(result.document)
                item_id = self.item_ids[text_index]
                item = self.items[item_id]
                
                recommendations.append({
                    'item': item,
                    'relevance': result.matching_score
                })
            
            return recommendations
        
        def find_similar_content(self, item_id: str, count: int = 5):
            """Find content similar to a specific item"""
            if item_id not in self.items:
                return []
            
            item = self.items[item_id]
            query = f"{item['title']} {item['description']} {' '.join(item['tags'])}"
            
            results = self.index.query(query, top_k=count + 1)  # +1 to exclude self
            
            similar_items = []
            for result in results:
                text_index = self.searchable_texts.index(result.document)
                similar_id = self.item_ids[text_index]
                
                # Skip the item itself
                if similar_id != item_id:
                    similar_items.append({
                        'item': self.items[similar_id],
                        'similarity': result.matching_score
                    })
            
            return similar_items[:count]

    # Usage
    content = [
        {
            'id': '1',
            'title': 'Introduction to Machine Learning',
            'description': 'Learn the basics of ML algorithms and applications',
            'tags': ['ml', 'python', 'beginner', 'tutorial']
        },
        {
            'id': '2', 
            'title': 'Advanced Deep Learning',
            'description': 'Dive deep into neural networks and deep learning',
            'tags': ['deep-learning', 'neural-networks', 'advanced', 'ai']
        },
        {
            'id': '3',
            'title': 'Python for Data Science',
            'description': 'Using Python libraries for data analysis',
            'tags': ['python', 'data-science', 'pandas', 'numpy']
        }
    ]
    
    recommender = ContentRecommender(content)
    
    # Get recommendations based on interests
    recommendations = recommender.recommend_by_interest("machine learning python")
    for rec in recommendations:
        print(f"Title: {rec['item']['title']}")
        print(f"Relevance: {rec['relevance']:.1f}%")
        print(f"Description: {rec['item']['description']}\n")

Performance Optimization Example
--------------------------------

Optimizing YARP for large datasets:

.. code-block:: python

    from yarp.vector_index import LocalMemoryIndex
    import time

    class OptimizedIndex:
        def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
            self.index = LocalMemoryIndex(documents, model_name)
        
        def build_optimized(self, num_trees=512):
            """Build index with more trees for better accuracy"""
            start_time = time.time()
            self.index.process(num_trees=num_trees)
            build_time = time.time() - start_time
            print(f"Index built in {build_time:.2f} seconds with {num_trees} trees")
        
        def benchmark_search(self, queries, search_k_values=[50, 100, 200]):
            """Benchmark different search_k values"""
            results = {}
            
            for search_k in search_k_values:
                total_time = 0
                for query in queries:
                    start_time = time.time()
                    self.index.query(query, search_k=search_k, top_k=5)
                    total_time += time.time() - start_time
                
                avg_time = total_time / len(queries)
                results[search_k] = avg_time
                print(f"search_k={search_k}: {avg_time*1000:.2f}ms per query")
            
            return results
        
        def memory_efficient_batch_add(self, documents, batch_size=1000):
            """Add documents in batches to manage memory usage"""
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.index.add(batch)
                print(f"Added batch {i//batch_size + 1}, total docs: {len(self.index.documents)}")

    # Usage for large datasets
    large_documents = [f"Document {i} with content..." for i in range(10000)]
    
    opt_index = OptimizedIndex(large_documents[:5000])  # Start with subset
    opt_index.build_optimized(num_trees=256)
    
    # Add remaining documents in batches
    opt_index.memory_efficient_batch_add(large_documents[5000:], batch_size=500)
    
    # Benchmark performance
    test_queries = ["content search", "document retrieval", "information finder"]
    opt_index.benchmark_search(test_queries)