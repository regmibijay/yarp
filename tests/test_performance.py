"""
Performance and benchmark tests for yarp package.
These tests measure performance characteristics and may be slow.
"""
import pytest
import time
import tempfile
import shutil
import os
from statistics import mean, median

from yarp.vector_index.local_vector_index import LocalMemoryIndex


@pytest.mark.slow
class TestPerformance:
    """Performance tests for LocalMemoryIndex operations."""
    
    def setup_method(self):
        """Set up for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create different document sets for testing
        self.small_docs = [
            f"Small document {i} with some content"
            for i in range(10)
        ]
        
        self.medium_docs = [
            f"Medium length document {i} with more substantial content "
            f"that includes multiple sentences and various topics related "
            f"to document {i} processing"
            for i in range(100)
        ]
        
        self.large_docs = [
            f"Large document {i} contains extensive text with detailed "
            f"information about topic {i}, including comprehensive analysis, "
            f"multiple paragraphs of content, various technical terms, "
            f"and substantial amounts of textual data that would be typical "
            f"in real-world document processing scenarios for document {i}"
            for i in range(500)
        ]
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_processing_performance_scaling(self):
        """Test how processing time scales with document count."""
        results = {}
        
        for name, docs in [
            ("small", self.small_docs),
            ("medium", self.medium_docs[:50]),  # Limit for CI/testing
            ("large", self.large_docs[:100])    # Limit for CI/testing
        ]:
            index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")
            
            start_time = time.time()
            index.process()
            end_time = time.time()
            
            processing_time = end_time - start_time
            results[name] = {
                'time': processing_time,
                'doc_count': len(docs),
                'time_per_doc': processing_time / len(docs)
            }
        
        # Basic performance assertions
        assert all(result['time'] > 0 for result in results.values())
        
        # Processing should complete in reasonable time
        for name, result in results.items():
            assert result['time'] < 60, f"{name} processing took too long: {result['time']:.2f}s"
    
    def test_query_performance(self):
        """Test query performance with different parameters."""
        # Use medium-sized dataset
        docs = self.medium_docs[:50]
        index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        # Test different query scenarios
        query_tests = [
            {"name": "small_k", "top_k": 5, "search_k": 50},
            {"name": "medium_k", "top_k": 20, "search_k": 100},
            {"name": "large_k", "top_k": 50, "search_k": 200},
        ]
        
        query_times = {}
        
        for test in query_tests:
            times = []
            
            # Run multiple queries to get average time
            for _ in range(5):
                start_time = time.time()
                results = index.query(
                    "test query document processing",
                    top_k=test["top_k"],
                    search_k=test["search_k"]
                )
                end_time = time.time()
                
                query_time = end_time - start_time
                times.append(query_time)
                
                # Verify results are correct
                assert len(results.results) <= test["top_k"]
            
            query_times[test["name"]] = {
                'mean': mean(times),
                'median': median(times),
                'times': times
            }
        
        # All queries should complete quickly
        for name, stats in query_times.items():
            assert stats['mean'] < 1.0, f"{name} queries too slow: {stats['mean']:.3f}s average"
    
    def test_add_performance(self):
        """Test performance of adding documents to existing index."""
        # Start with base documents
        base_docs = self.small_docs
        index = LocalMemoryIndex(base_docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        # Test adding different numbers of documents
        add_tests = [
            {"name": "single", "docs": ["New single document"]},
            {"name": "batch_small", "docs": [f"New doc {i}" for i in range(5)]},
            {"name": "batch_medium", "docs": [f"New doc {i}" for i in range(20)]},
        ]
        
        add_times = {}
        
        for test in add_tests:
            start_time = time.time()
            index.add(test["docs"])
            end_time = time.time()
            
            add_time = end_time - start_time
            add_times[test["name"]] = {
                'time': add_time,
                'doc_count': len(test["docs"]),
                'time_per_doc': add_time / len(test["docs"])
            }
        
        # Adding documents should complete in reasonable time
        for name, stats in add_times.items():
            assert stats['time'] < 10, f"{name} add operation too slow: {stats['time']:.2f}s"
    
    def test_backup_load_performance(self):
        """Test backup and load operation performance."""
        # Create index with moderate amount of data
        docs = self.medium_docs[:30]
        index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        backup_path = os.path.join(self.temp_dir, "perf_backup")
        
        # Test backup performance
        start_time = time.time()
        index.backup(backup_path)
        backup_time = time.time() - start_time
        
        # Test load performance
        start_time = time.time()
        loaded_index = LocalMemoryIndex.load(
            backup_path,
            model_name="all-MiniLM-L6-v2"
        )
        load_time = time.time() - start_time
        
        # Verify loaded index works
        results = loaded_index.query("test", top_k=5)
        assert len(results.results) > 0
        
        # Performance assertions
        assert backup_time < 10, f"Backup too slow: {backup_time:.2f}s"
        assert load_time < 10, f"Load too slow: {load_time:.2f}s"
        
        return {
            'backup_time': backup_time,
            'load_time': load_time,
            'doc_count': len(docs)
        }
    
    def test_memory_usage_patterns(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and process multiple indices
        indices = []
        for i in range(3):
            docs = [f"Document set {i} doc {j}" for j in range(20)]
            index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")
            index.process()
            indices.append(index)
        
        # Check memory after creating indices
        after_creation_memory = process.memory_info().rss
        memory_increase = after_creation_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for test data)
        assert memory_increase < 500 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"
        
        # Test queries on all indices
        for i, index in enumerate(indices):
            results = index.query(f"document set {i}", top_k=5)
            assert len(results.results) > 0
        
        # Clean up indices
        del indices
        
        final_memory = process.memory_info().rss
        
        return {
            'initial_memory_mb': initial_memory / 1024 / 1024,
            'peak_memory_mb': after_creation_memory / 1024 / 1024,
            'final_memory_mb': final_memory / 1024 / 1024,
            'memory_increase_mb': memory_increase / 1024 / 1024
        }
    
    def test_concurrent_query_performance(self):
        """Test performance with multiple concurrent queries."""
        import threading
        import queue
        
        docs = self.medium_docs[:30]
        index = LocalMemoryIndex(docs, model_name="all-MiniLM-L6-v2")
        index.process()
        
        # Results queue for threads
        results_queue = queue.Queue()
        
        def query_worker(worker_id, num_queries):
            """Worker function for concurrent queries."""
            times = []
            for i in range(num_queries):
                start_time = time.time()
                result = index.query(
                    f"worker {worker_id} query {i}",
                    top_k=10
                )
                end_time = time.time()
                
                times.append(end_time - start_time)
                assert len(result.results) <= 10
            
            results_queue.put({
                'worker_id': worker_id,
                'times': times,
                'mean_time': mean(times)
            })
        
        # Run concurrent queries
        num_workers = 3
        queries_per_worker = 5
        
        threads = []
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=query_worker,
                args=(worker_id, queries_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        worker_results = []
        while not results_queue.empty():
            worker_results.append(results_queue.get())
        
        assert len(worker_results) == num_workers
        
        # Performance assertions
        assert total_time < 20, f"Concurrent queries too slow: {total_time:.2f}s"
        
        for result in worker_results:
            assert result['mean_time'] < 2.0, f"Worker {result['worker_id']} queries too slow: {result['mean_time']:.2f}s"
        
        return {
            'total_time': total_time,
            'worker_results': worker_results,
            'total_queries': num_workers * queries_per_worker
        }


@pytest.mark.slow  
class TestScalability:
    """Test scalability with larger datasets."""
    
    @pytest.mark.skipif(
        os.getenv('SKIP_LARGE_TESTS', 'false').lower() == 'true',
        reason="Large tests skipped by environment variable"
    )
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create a substantial dataset
        large_docs = [
            f"Large scale document {i} containing substantial content "
            f"with multiple topics, detailed information, and comprehensive "
            f"text that simulates real-world document processing scenarios. "
            f"This document covers topic area {i % 10} and includes "
            f"technical details, analysis, and extensive textual content."
            for i in range(1000)  # 1000 documents
        ]
        
        index = LocalMemoryIndex(
            large_docs,
            model_name="all-MiniLM-L6-v2"
        )
        
        # Process should complete without errors
        start_time = time.time()
        index.process(num_trees=64)  # Fewer trees for faster processing
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert processing_time < 180, f"Large dataset processing too slow: {processing_time:.1f}s"
        
        # Test querying
        start_time = time.time()
        results = index.query("large scale document technical", top_k=20)
        query_time = time.time() - start_time
        
        assert len(results.results) == 20
        assert query_time < 2.0, f"Large dataset query too slow: {query_time:.2f}s"
        
        return {
            'doc_count': len(large_docs),
            'processing_time': processing_time,
            'query_time': query_time,
            'embeddings_shape': index.embeddings.shape
        }