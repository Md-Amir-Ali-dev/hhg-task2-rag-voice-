import asyncio
import time
import argparse
import json
import os
import numpy as np
from dotenv import load_dotenv

from stt import SarvamSTTClient
from retriever import FastRetriever
from llm import FastLLM

# Load environment variables
load_dotenv()


class VoiceRAGPipeline:
    def __init__(self, use_mock=False):
        self.stt_client = SarvamSTTClient(use_mock=use_mock)
        self.retriever = FastRetriever()
        self.llm_client = FastLLM(use_mock=use_mock)

    async def process_audio_stream(self, audio_generator):
        """
        End-to-end pipeline: STT → Retrieval → LLM Generation.
        """
        start_e2e = time.time()

        # 1. Speech-to-Text (Sarvam)
        print("Transcribing audio via Sarvam STT...")
        transcript = await self.stt_client.transcribe_stream(audio_generator)
        stt_latency = time.time() - start_e2e
        print(f"Transcript: '{transcript}' (Latency: {stt_latency*1000:.1f}ms)")

        # 2. Retrieval
        print("Retrieving context...")
        retrieval_result = self.retriever.retrieve(transcript)
        print(f"Retrieved Context (Latency: {retrieval_result.latency*1000:.1f}ms)")
        if retrieval_result.unsafe:
            print("-> Guardrail triggered: Query unsafe/inappropriate.")
        elif retrieval_result.off_topic:
            print("-> Guardrail triggered: Query off-topic or low confidence.")

        # 3. Generation (with harness)
        print("Generating answer with harness...")
        generation_result = await self.llm_client.generate_answer(
            query=transcript,
            context=retrieval_result.context,
            off_topic=retrieval_result.off_topic,
            unsafe=retrieval_result.unsafe
        )

        end_e2e = time.time()
        total_latency = end_e2e - start_e2e

        metrics = {
            "stt_ms": round(stt_latency * 1000, 1),
            "retrieval_ms": round(retrieval_result.latency * 1000, 1),
            "llm_ttft_ms": round(generation_result.ttft * 1000, 1),
            "llm_total_ms": round(generation_result.latency * 1000, 1),
            "total_e2e_ms": round(total_latency * 1000, 1),
            "best_sim_score": round(float(retrieval_result.best_score), 3),
            "retries_used": generation_result.retries_used
        }

        return {
            "transcript": transcript,
            "answer": generation_result.answer,
            "grounded": generation_result.grounded,
            "confidence_note": generation_result.confidence_note,
            "chunk_sources": retrieval_result.chunk_sources,
            "metrics": metrics
        }

    async def process_text_query(self, query: str):
        """Processes text query directly (skipping STT)."""
        start_e2e = time.time()
        retrieval_result = self.retriever.retrieve(query)
        
        generation_result = await self.llm_client.generate_answer(
            query=query,
            context=retrieval_result.context,
            off_topic=retrieval_result.off_topic,
            unsafe=retrieval_result.unsafe
        )
        total_latency = time.time() - start_e2e

        metrics = {
            "stt_ms": 0.0,
            "retrieval_ms": round(retrieval_result.latency * 1000, 1),
            "llm_ttft_ms": round(generation_result.ttft * 1000, 1),
            "llm_total_ms": round(generation_result.latency * 1000, 1),
            "total_e2e_ms": round(total_latency * 1000, 1),
            "best_sim_score": round(float(retrieval_result.best_score), 3),
            "retries_used": generation_result.retries_used
        }

        return {
            "query": query,
            "answer": generation_result.answer,
            "grounded": generation_result.grounded,
            "confidence_note": generation_result.confidence_note,
            "chunk_sources": retrieval_result.chunk_sources,
            "off_topic": retrieval_result.off_topic or retrieval_result.unsafe,
            "metrics": metrics
        }


async def mock_audio_generator():
    """Yields mock audio chunks to simulate streaming."""
    for _ in range(10):
        yield b'\x00' * 1024
        await asyncio.sleep(0.005)


async def run_benchmark(num_runs=30, use_mock=True):
    print(f"\n=======================================================")
    print(f"  RUNNING VOICE RAG BENCHMARK ({num_runs} queries | Mock={use_mock})")
    print(f"=======================================================\n")
    
    pipeline = VoiceRAGPipeline(use_mock=use_mock)

    test_queries = [
        "What is the capital of India?",
        "Who invented the telephone?",
        "What is machine learning?",
        "Explain photosynthesis briefly",
        "Who built the Taj Mahal?",
        "How do neural networks work?",
        "What is the population of New Delhi?",
        "Tell me about Indian history and culture",
        "What is artificial intelligence?",
        "How do solar panels generate electricity?"
    ]

    total_latencies = []
    retrieval_latencies = []
    llm_latencies = []
    llm_ttfts = []

    for i in range(num_runs):
        query = test_queries[i % len(test_queries)]
        print(f"[{i+1}/{num_runs}] Query: '{query}'")
        result = await pipeline.process_text_query(query)
        
        m = result['metrics']
        total_latencies.append(m['total_e2e_ms'])
        retrieval_latencies.append(m['retrieval_ms'])
        llm_latencies.append(m['llm_total_ms'])
        llm_ttfts.append(m['llm_ttft_ms'])

    # Compute percentiles
    p50 = float(np.percentile(total_latencies, 50))
    p70 = float(np.percentile(total_latencies, 70))
    p95 = float(np.percentile(total_latencies, 95))
    p100 = float(np.percentile(total_latencies, 100))

    retrieval_p50 = float(np.percentile(retrieval_latencies, 50))
    retrieval_p95 = float(np.percentile(retrieval_latencies, 95))
    
    llm_p50 = float(np.percentile(llm_latencies, 50))
    llm_p95 = float(np.percentile(llm_latencies, 95))

    report = {
        "num_runs": num_runs,
        "use_mock": use_mock,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_target_ms": 200.0,
        "meets_target_p50": p50 <= 200.0,
        "meets_target_p70": p70 <= 200.0,
        "p50_ms": round(p50, 1),
        "p70_ms": round(p70, 1),
        "p95_ms": round(p95, 1),
        "p100_ms": round(p100, 1),
        "breakdown": {
            "retrieval_p50_ms": round(retrieval_p50, 1),
            "retrieval_p95_ms": round(retrieval_p95, 1),
            "llm_p50_ms": round(llm_p50, 1),
            "llm_p95_ms": round(llm_p95, 1),
            "llm_ttft_p50_ms": round(float(np.percentile(llm_ttfts, 50)), 1)
        },
        "all_latencies_ms": [round(x, 1) for x in total_latencies]
    }

    # Save to file
    with open("latency_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=======================================================")
    print("                 LATENCY ANALYTICS REPORT              ")
    print("=======================================================")
    print(f" Target Latency:  < 200 ms")
    print(f" P50 Latency:     {p50:.1f} ms  {'[PASS]' if p50 <= 200 else '[FAIL]'}")
    print(f" P70 Latency:     {p70:.1f} ms  {'[PASS]' if p70 <= 200 else '[FAIL]'}")
    print(f" P95 Latency:     {p95:.1f} ms")
    print(f" P100 Latency:    {p100:.1f} ms")
    print("-------------------------------------------------------")
    print(f" Retrieval P50:   {retrieval_p50:.1f} ms | P95: {retrieval_p95:.1f} ms")
    print(f" LLM Gen P50:     {llm_p50:.1f} ms | P95: {llm_p95:.1f} ms")
    print("=======================================================")
    print("Saved report to latency_report.json\n")
    return report


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run latency analytics benchmark")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark queries")
    parser.add_argument("--mock", action="store_true", help="Use mock STT and LLM")
    args = parser.parse_args()

    if args.benchmark:
        await run_benchmark(num_runs=args.runs, use_mock=args.mock)
    else:
        pipeline = VoiceRAGPipeline(use_mock=args.mock)
        print("Running single test query...")
        result = await pipeline.process_audio_stream(mock_audio_generator())
        print("\n--- Final Output ---")
        print(f"Q: {result['transcript']}")
        print(f"A: {result['answer']}")
        print(f"Grounded: {result['grounded']} ({result['confidence_note']})")
        print(f"Total Time: {result['metrics']['total_e2e_ms']:.1f} ms")
        print("Metrics Breakdown:")
        for k, v in result['metrics'].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
