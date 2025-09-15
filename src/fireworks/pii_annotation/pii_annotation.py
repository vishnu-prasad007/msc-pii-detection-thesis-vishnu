import json
import os
import asyncio
import sys
import psutil
import time
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from labelrix import segment_lines, annotation_filters

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    io_wait: float

class DynamicConcurrencyManager:
    def __init__(self, initial_concurrency: int = 5, min_concurrency: int = 1, max_concurrency: int = 20):
        self.current_concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.rate_limit_remaining = self.max_concurrency  # Start with max_concurrency as default
        self.rate_limit_reset = 60  # default 60s window
        self.adjustment_interval = 10  # seconds between adjustments
        self._active_tasks: Set[str] = set()  # Track active task IDs
        self.lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()  # Separate lock for task management
        
        # Initialize token bucket for concurrency control
        self.available_tokens = initial_concurrency
        self.token_event = asyncio.Event()
        self.token_event.set()
        
        # For background monitoring
        self._stop_monitoring = False
        self._monitor_task = None

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource usage"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            io_wait=psutil.cpu_times_percent(interval=1).iowait
        )

    async def update_rate_limits(self, headers: Dict[str, str]) -> None:
        """Update rate limits based on API response headers"""
        async with self.lock:
            remaining = headers.get('x-ratelimit-remaining')
            if remaining is not None:
                self.rate_limit_remaining = int(remaining)
            
            reset = headers.get('x-ratelimit-reset')
            if reset is not None:
                self.rate_limit_reset = int(float(reset))

    async def acquire_token(self, task_id: str) -> None:
        """Acquire a token for task execution"""
        while True:
            async with self.lock:
                if self.available_tokens > 0:
                    self.available_tokens -= 1
                    async with self._task_lock:
                        self._active_tasks.add(task_id)
                    if self.available_tokens == 0:
                        self.token_event.clear()
                    return
            await self.token_event.wait()

    async def release_token(self, task_id: str) -> None:
        """Release a token after task completion"""
        async with self.lock:
            self.available_tokens += 1
            self.token_event.set()
            async with self._task_lock:
                self._active_tasks.discard(task_id)

    async def adjust_concurrency(self) -> None:
        """Adjust concurrency based on system metrics and rate limits"""
        async with self.lock:
            metrics = self.get_system_metrics()
            
            # Start with current concurrency
            new_concurrency = self.current_concurrency

            # Adjust based on system metrics
            if metrics.cpu_percent > 80 or metrics.memory_percent > 80 or metrics.io_wait > 20:
                new_concurrency = max(self.min_concurrency, new_concurrency - 1)
            elif metrics.cpu_percent < 50 and metrics.memory_percent < 50 and metrics.io_wait < 10:
                new_concurrency = min(self.max_concurrency, new_concurrency + 1)

            # Adjust based on rate limits
            if self.rate_limit_remaining < new_concurrency:
                new_concurrency = max(self.min_concurrency, self.rate_limit_remaining)

            # Update concurrency if changed
            if new_concurrency != self.current_concurrency:
                print(f"Adjusting concurrency from {self.current_concurrency} to {new_concurrency}")
                print(f"System metrics - CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%, IO Wait: {metrics.io_wait}%")
                print(f"Rate limit remaining: {self.rate_limit_remaining}")
                print(f"Active tasks: {len(self._active_tasks)}")
                
                # Update the concurrency limit
                self.current_concurrency = new_concurrency
                
                # Adjust available tokens while respecting active tasks
                async with self._task_lock:
                    active_count = len(self._active_tasks)
                    if active_count > new_concurrency:
                        # If we have more active tasks than new limit, let them finish
                        self.available_tokens = 0
                        self.token_event.clear()
                    else:
                        # Otherwise, adjust available tokens
                        self.available_tokens = new_concurrency - active_count
                        if self.available_tokens > 0:
                            self.token_event.set()
                        else:
                            self.token_event.clear()

    async def _monitor_metrics(self) -> None:
        """Background task to periodically check metrics and adjust concurrency"""
        while not self._stop_monitoring:
            await self.adjust_concurrency()
            await asyncio.sleep(self.adjustment_interval)

    async def start_monitoring(self) -> None:
        """Start the background monitoring task"""
        self._stop_monitoring = False
        self._monitor_task = asyncio.create_task(self._monitor_metrics())

    async def stop_monitoring(self) -> None:
        """Stop the background monitoring task"""
        if self._monitor_task:
            self._stop_monitoring = True
            await self._monitor_task
            self._monitor_task = None

def process_file_sync(file_path: str, pages: List[int], model: str, votes_out_dir: str) -> None:
    """Synchronous version of process_file for thread pool"""
    try:
        print(f"Started processing file {file_path}")
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Extract votes using LLMs
        loop.run_until_complete(
            segment_lines.extract_votes(file_path, pages, model, out_dir=votes_out_dir)
        )
        print(f"Completed processing file {file_path}")
        
    except Exception as e:
        print(f"Failed to process file {file_path}: {str(e)}")
    finally:
        loop.close()

async def main():
    MODEL = "Qwen/Qwen3-30B-A3B"
    MODEL_DIR = "qwen3-30b-a3b"
    benchmark_file_list_path = "/root/annotations/data/ad-buy-forms/top_files_ad_buy_forms-v1.json"
    original_tar_json_file_dir = "/root/annotations/"
    out_path = "/root/annotations/labels/ad-buy-forms/"

    try:
        with open(benchmark_file_list_path, "r") as benchmark_file_list_file:
            data = json.load(benchmark_file_list_file)
        
        votes_out_dir = out_path + MODEL_DIR + "-per_page_votes"
        votes_merged_dir = out_path + MODEL_DIR + "-per_page_votes_merged"
        os.makedirs(votes_out_dir, exist_ok=True)
        os.makedirs(votes_merged_dir, exist_ok=True)

        # Here we filter already processed files
        processed_files = os.listdir(votes_out_dir)
        # Extract doc IDs from processed files (e.g., "votes_ffbc0084_page1.csv" -> "ffbc0084")
        processed_doc_ids = {f.split('_')[1].split('.')[0] for f in processed_files if f.startswith('votes_')}
        print(len(processed_doc_ids))
        # Filter data based on doc IDs
        data = [item for item in data 
               if os.path.basename(item['train_source_path']).split('.')[0] not in processed_doc_ids]

        print(f"Found {len(data)} files to process")

        # data = data[:10]

        # Create thread pool
        max_workers = 50  # Use half of available CPU cores (16 cores total)
        total_files = len(data)
        processed = 0

        print(f"\nStarting processing of {total_files} files with {max_workers} concurrent workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files to the thread pool
            future_to_file = {
                executor.submit(
                    process_file_sync,
                    os.path.join(original_tar_json_file_dir, item['train_source_path']),
                    [item['page_number']],
                    MODEL,
                    votes_out_dir
                ): item['train_source_path']
                for item in data
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    future.result()  # This will raise any exceptions that occurred
                    processed += 1
                    print(f"\nProgress: {processed}/{total_files} files processed")
                    print(f"Completed: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        print("\nAll files processed successfully!")
        
        # Here we apply filtering
        print("\nApplying filtering...")
        annotation_filters.process_directory_to_json(
            in_dir=votes_out_dir,
            out_dir=votes_merged_dir,
            iou_thresh=0.5,
            overlap_thresh=0.9
        )
        print("Filtering complete!")
        
    finally:
        print("\nProcessing complete!")

if __name__ == "__main__":
    asyncio.run(main())
