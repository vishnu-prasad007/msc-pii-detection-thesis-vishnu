#!/usr/bin/env python3
"""
Comprehensive Model Benchmarking Script

This script benchmarks all available models on PII extraction tasks,
providing detailed comparisons across different entity types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from compute_overall_metrics import compute_overall_metrics

warnings.filterwarnings('ignore')

# Set pandas display options for better formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 4)


 # Define all model paths
models = {
        # 'Qwen3-30B-A3B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Qwen3-30B-A3B/",
        'Qwen3-30B-A3B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Qwen3-30B-A3B-new-prompts-1/",
        'Qwen3-4B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Qwen3-4B/",
        'Qwen2.5-32B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Qwen2.5-32B/",
        'Qwen2.5-72B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Qwen2.5-72B/",
        'Llama-3.3-70B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Llama-3.3-70B/",
        'Gemma-3-4B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/gemma-3-4b-it/",
        'Gemma-3-27B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/gemma-3-27b-it/"
    }

# Udate to below for AD-Buy
# models = {
#         'Qwen3-30B-A3B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/AD_BUY_AUTO_LABELS/Qwen3-30B-A3B-per_page_votes_merged/",
#         'Qwen3-4B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/AD_BUY_AUTO_LABELS/Qwen3-4B-fast-per_page_votes_merged/",
#         'Qwen2.5-32B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/AD_BUY_AUTO_LABELS/Qwen2.5-32B-Instruct-fast-per_page_votes_merged/",
#         'Qwen2.5-72B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/AD_BUY_AUTO_LABELS/Qwen2.5-72B-Instruct-per_page_votes_merged/",
#         'Llama-3.3-70B': "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/AD_BUY_AUTO_LABELS/Llama-3.3-70B-Instruct-per_page_votes_merged/",
#     }

ground_truth_dir = "/Volumes/MyDataDrive/thesis/code-2/data/test/labels/"

# ground_truth_dir = "/Volumes/MyDataDrive/thesis/code-2/src/labelrix/ad-buy-form-test-labels/"

def main():

    print(f"Found {len(models)} models to benchmark:")
    for model_name in models.keys():
        print(f"  - {model_name}")

    # Run the benchmark
    print("\nStarting comprehensive model benchmark...")
    benchmark_results = benchmark_all_models(models, ground_truth_dir)
    print("✓ Benchmark completed!")

    # Create tables
    overall_table, detailed_table = create_benchmark_tables(benchmark_results)
    precision_table, recall_table, f1_table = create_pivot_tables(detailed_table)
    ranking_table = create_ranking_table(f1_table)

    # Display results with pandas formatting
    display_results_pandas(overall_table, precision_table, recall_table, f1_table, ranking_table)

    # Save results
    # save_results(overall_table, detailed_table, precision_table, recall_table, f1_table, ranking_table)

    # Print summary
    print_summary(overall_table, f1_table)

def benchmark_all_models(models, ground_truth_dir, verbose=True):
    """Benchmark all models and return results in a structured format."""
    all_results = {}
    
    for i, (model_name, predictions_dir) in enumerate(models.items(), 1):
        if verbose:
            print(f"\n[{i}/{len(models)}] Evaluating {model_name}...")
        
        try:
            results = compute_overall_metrics(
                ground_truth_dir=ground_truth_dir,
                predictions_dir=predictions_dir,
                verbose=False
            )
            all_results[model_name] = results

            if verbose:
                print(f"  ✓ Macro F1: {results['overall']['f1']:.4f}")
                
        except Exception as e:
            print(f"  ✗ Error evaluating {model_name}: {e}")
            all_results[model_name] = None
    
    return all_results

def create_benchmark_tables(benchmark_results):
    """Create comprehensive benchmark tables from the results."""
    overall_data = []
    detailed_data = []
    
    for model_name, results in benchmark_results.items():
        if results is None:
            continue
            
        # Overall metrics
        overall_data.append({
            'Model': model_name,
            'Macro Precision': results['overall']['precision'],
            'Macro Recall': results['overall']['recall'],
            'Macro F1': results['overall']['f1'],
            'True Positives': results['overall']['tp'],
            'False Positives': results['overall']['fp'],
            'False Negatives': results['overall']['fn']
        })
        
        # Per-entity-type metrics
        for entity_type, metrics in results['per_entity_type'].items():
            print(entity_type)
            detailed_data.append({
                'Model': model_name,
                'Entity Type': entity_type,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'True Positives': metrics['tp'],
                'False Positives': metrics['fp'],
                'False Negatives': metrics['fn'],
                'Ground Truth Count': metrics['ground_truth_count'],
                'Predicted Count': metrics['predicted_count']
            })
            
    overall_df = pd.DataFrame(overall_data)
    detailed_df = pd.DataFrame(detailed_data)
    
    return overall_df, detailed_df

def create_pivot_tables(detailed_table):
    """Create pivot tables for easy comparison across models for each metric."""
    precision_pivot = detailed_table.pivot(index='Entity Type', columns='Model', values='Precision')
    recall_pivot = detailed_table.pivot(index='Entity Type', columns='Model', values='Recall')
    f1_pivot = detailed_table.pivot(index='Entity Type', columns='Model', values='F1 Score')
    
    return precision_pivot, recall_pivot, f1_pivot

def create_ranking_table(f1_table):
    ranking_rows = []
    for entity_type in f1_table.index:
        scores = f1_table.loc[entity_type].dropna()
        sorted_scores = scores.sort_values(ascending=False)
        labels = [f"{m} ({v:.4f})" for m, v in sorted_scores.items()]

        def get(i): 
            return labels[i] if i < len(labels) else "-"

        worst = labels[-1] if labels else "-"
        ranking_rows.append({
            "Entity Type": entity_type,
            "1st Place": get(0),
            "2nd Place": get(1),
            "3rd Place": get(2),
            "Worst": worst,
        })
    return pd.DataFrame(ranking_rows)

def display_results_pandas(overall_table, precision_table, recall_table, f1_table, ranking_table):
    """Display results using pandas with better formatting."""
    
    print("\n" + "="*100)
    print("OVERALL MODEL BENCHMARK RESULTS (Macro-Averaged)")
    print("="*100)
    
    # Format overall table for better display
    overall_display = overall_table.copy()
    overall_display['Macro Precision'] = overall_display['Macro Precision'].map('{:.4f}'.format)
    overall_display['Macro Recall'] = overall_display['Macro Recall'].map('{:.4f}'.format)
    overall_display['Macro F1'] = overall_display['Macro F1'].map('{:.4f}'.format)
    
    print(overall_display.to_string(index=False))
    
    print("\n" + "="*100)
    print("PRECISION COMPARISON BY ENTITY TYPE")
    print("="*100)
    print(precision_table.round(4).to_string())
    
    print("\n" + "="*100)
    print("RECALL COMPARISON BY ENTITY TYPE")
    print("="*100)
    print(recall_table.round(4).to_string())
    
    print("\n" + "="*100)
    print("F1 SCORE COMPARISON BY ENTITY TYPE")
    print("="*100)
    print(f1_table.round(4).to_string())
    
    print("\n" + "="*100)
    print("MODEL RANKINGS BY ENTITY TYPE")
    print("="*100)
    print(ranking_table.to_string(index=False))
    
    # Create a summary table with best performers
    print("\n" + "="*100)
    print("BEST PERFORMING MODELS BY ENTITY TYPE")
    print("="*100)
    
    best_models_summary = []
    for entity_type in f1_table.index:
        scores = f1_table.loc[entity_type]
        best_model = scores.idxmax()
        best_score = scores.max()
        best_models_summary.append({
            'Entity Type': entity_type,
            'Best Model': best_model,
            'F1 Score': f"{best_score:.4f}",
            'Precision': f"{precision_table.loc[entity_type, best_model]:.4f}",
            'Recall': f"{recall_table.loc[entity_type, best_model]:.4f}"
        })
    
    best_summary_df = pd.DataFrame(best_models_summary)
    print(best_summary_df.to_string(index=False))

def save_results(overall_table, detailed_table, precision_table, recall_table, f1_table, ranking_table):
    """Save all results to CSV files."""
    overall_table.to_csv('benchmark_overall_results.csv', index=False)
    detailed_table.to_csv('benchmark_detailed_results.csv', index=False)
    precision_table.to_csv('benchmark_precision_by_entity.csv')
    recall_table.to_csv('benchmark_recall_by_entity.csv')
    f1_table.to_csv('benchmark_f1_by_entity.csv')
    ranking_table.to_csv('benchmark_rankings.csv', index=False)
    
    print("\n" + "="*100)
    print("RESULTS SAVED TO CSV FILES")
    print("="*100)
    print("- benchmark_overall_results.csv")
    print("- benchmark_detailed_results.csv")
    print("- benchmark_precision_by_entity.csv")
    print("- benchmark_recall_by_entity.csv")
    print("- benchmark_f1_by_entity.csv")
    print("- benchmark_rankings.csv")

def print_summary(overall_table, f1_table):
    """Print summary statistics."""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    print(f"\nNumber of models evaluated: {len(overall_table)}")
    print(f"\nMacro F1 Score Statistics:")
    print(f"  Mean: {overall_table['Macro F1'].mean():.4f}")
    print(f"  Std:  {overall_table['Macro F1'].std():.4f}")
    print(f"  Min:  {overall_table['Macro F1'].min():.4f}")
    print(f"  Max:  {overall_table['Macro F1'].max():.4f}")

    best_overall_idx = overall_table['Macro F1'].idxmax()
    print(f"\nBest Overall Model: {overall_table.loc[best_overall_idx, 'Model']}")
    print(f"Best Macro F1: {overall_table.loc[best_overall_idx, 'Macro F1']:.4f}")
    print(f"Best Macro Precision: {overall_table.loc[best_overall_idx, 'Macro Precision']:.4f}")
    print(f"Best Macro Recall: {overall_table.loc[best_overall_idx, 'Macro Recall']:.4f}")

    # Create a summary DataFrame
    summary_data = []
    for entity_type in f1_table.index:
        scores = f1_table.loc[entity_type]
        best_model = scores.idxmax()
        best_score = scores.max()
        worst_model = scores.idxmin()
        worst_score = scores.min()
        mean_score = scores.mean()
        
        summary_data.append({
            'Entity Type': entity_type,
            'Best Model': best_model,
            'Best F1': f"{best_score:.4f}",
            'Worst Model': worst_model,
            'Worst F1': f"{worst_score:.4f}",
            'Mean F1': f"{mean_score:.4f}",
            'Std F1': f"{scores.std():.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\nDetailed Entity Type Summary:")
    print(summary_df.to_string(index=False))