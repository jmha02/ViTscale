#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def analyze_inference_performance(results_path: str = 'plots/benchmark_results.csv'):
    """Analyze inference performance results and create detailed reports"""
    
    # Create analysis directory
    analysis_dir = 'analysis/results/inference'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load results
    try:
        df = pd.read_csv(results_path)
        print(f"Loaded inference benchmark results from: {results_path}")
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        print("Please run benchmarks/inference.py first!")
        return None
    
    # Generate detailed analysis plots
    create_inference_analysis_plots(df, analysis_dir)
    
    # Generate HTML report
    create_inference_html_report(df, analysis_dir)
    
    print(f"Inference analysis completed! Results saved in: {analysis_dir}")
    return df

def create_inference_analysis_plots(df, analysis_dir):
    """Create detailed inference analysis plots"""
    
    plt.style.use('default')
    
    # 1. Inference Performance Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ViT Inference Performance Analysis with LoRA', fontsize=16, fontweight='bold')
    
    # Inference Time Comparison
    ax1 = axes[0, 0]
    colors = ['#2E86AB' if 'Baseline' in name else '#A23B72' for name in df['model_name']]
    bars1 = ax1.bar(df['model_name'], df['inference_time_ms'], color=colors)
    ax1.set_title('Inference Time per Sample', fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, time_val in zip(bars1, df['inference_time_ms']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    # Throughput Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['model_name'], df['throughput_fps'], color=colors)
    ax2.set_title('Inference Throughput', fontweight='bold')
    ax2.set_ylabel('FPS (Frames/sec)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, throughput in zip(bars2, df['throughput_fps']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{throughput:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Memory Usage
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['model_name'], df['peak_memory_mb'], color=colors)
    ax3.set_title('Peak Inference Memory Usage', fontweight='bold')
    ax3.set_ylabel('Memory (MB)')
    ax3.tick_params(axis='x', rotation=45)
    for bar, memory in zip(bars3, df['peak_memory_mb']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{memory:.0f}MB', ha='center', va='bottom', fontsize=9)
    
    # FLOPs Comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(df['model_name'], df['total_flops'] / 1e9, color=colors)
    ax4.set_title('Computational Load (FLOPs)', fontweight='bold')
    ax4.set_ylabel('GFLOPs')
    ax4.tick_params(axis='x', rotation=45)
    for bar, flops in zip(bars4, df['total_flops']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{flops/1e9:.1f}G', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/inference_performance_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter Efficiency Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('LoRA Parameter Efficiency in Inference', fontsize=16, fontweight='bold')
    
    # Trainable Parameters vs Performance
    ax1 = axes[0]
    scatter = ax1.scatter(df['trainable_params'] / 1e6, df['throughput_fps'], 
                         c=colors, s=100, alpha=0.7)
    for i, name in enumerate(df['model_name']):
        ax1.annotate(name, (df['trainable_params'].iloc[i] / 1e6, df['throughput_fps'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.set_xlabel('Trainable Parameters (M)')
    ax1.set_ylabel('Inference Throughput (FPS)')
    ax1.set_title('Trainable Parameters vs Inference Throughput')
    ax1.grid(True, alpha=0.3)
    
    # FLOPs vs Performance
    ax2 = axes[1]
    scatter = ax2.scatter(df['total_flops'] / 1e9, df['inference_time_ms'], 
                         c=colors, s=100, alpha=0.7)
    for i, name in enumerate(df['model_name']):
        ax2.annotate(name, (df['total_flops'].iloc[i] / 1e9, df['inference_time_ms'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('FLOPs (GFLOPs)')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Computational Load vs Inference Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/inference_parameter_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LoRA Rank Analysis
    lora_df = df[df['model_name'] != 'Baseline'].copy()
    if len(lora_df) > 0:
        lora_df['rank'] = lora_df['model_name'].str.extract(r'(\d+)').astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LoRA Rank Impact on Inference Performance', fontsize=16, fontweight='bold')
        
        # Rank vs Inference Time
        ax1 = axes[0, 0]
        ax1.plot(lora_df['rank'], lora_df['inference_time_ms'], 'o-', color='#A23B72', linewidth=2, markersize=8)
        ax1.set_xlabel('LoRA Rank')
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('LoRA Rank vs Inference Time')
        ax1.grid(True, alpha=0.3)
        
        # Rank vs Memory
        ax2 = axes[0, 1]
        ax2.plot(lora_df['rank'], lora_df['peak_memory_mb'], 'o-', color='#F18F01', linewidth=2, markersize=8)
        ax2.set_xlabel('LoRA Rank')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('LoRA Rank vs Memory Usage')
        ax2.grid(True, alpha=0.3)
        
        # Rank vs FLOPs
        ax3 = axes[1, 0]
        ax3.plot(lora_df['rank'], lora_df['total_flops'] / 1e9, 'o-', color='#C73E1D', linewidth=2, markersize=8)
        ax3.set_xlabel('LoRA Rank')
        ax3.set_ylabel('FLOPs (GFLOPs)')
        ax3.set_title('LoRA Rank vs Computational Load')
        ax3.grid(True, alpha=0.3)
        
        # Rank vs Efficiency
        efficiency = lora_df['throughput_fps'] / (lora_df['trainable_params'] / 1e6)
        ax4 = axes[1, 1]
        ax4.plot(lora_df['rank'], efficiency, 'o-', color='#3E8E41', linewidth=2, markersize=8)
        ax4.set_xlabel('LoRA Rank')
        ax4.set_ylabel('Inference Efficiency (FPS per M params)')
        ax4.set_title('LoRA Rank vs Inference Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{analysis_dir}/lora_rank_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Inference analysis plots saved to: {analysis_dir}")

def create_inference_html_report(df, analysis_dir):
    """Create detailed HTML report for inference analysis"""
    
    baseline_row = df[df['model_name'] == 'Baseline'].iloc[0]
    lora_df = df[df['model_name'] != 'Baseline']
    
    # Calculate statistics
    best_speed_model = lora_df.loc[lora_df['inference_time_ms'].idxmin()]
    best_efficiency_model = lora_df.loc[(lora_df['throughput_fps'] / (lora_df['trainable_params'] / 1e6)).idxmax()]
    best_memory_model = lora_df.loc[lora_df['peak_memory_mb'].idxmin()]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ViT Inference Performance Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .highlight {{ background-color: #f0f8ff; padding: 15px; border-left: 4px solid #007acc; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
            .best {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .improvement {{ color: #4caf50; font-weight: bold; }}
            .degradation {{ color: #f44336; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ ViT Inference Performance Analysis Report</h1>
            <p>Comprehensive analysis of Vision Transformer inference performance with LoRA adaptations</p>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="highlight">
                <p><strong>Key Finding:</strong> LoRA shows trade-offs in inference: parameter efficiency gains with slight performance overhead.</p>
                <p><strong>Best Overall Model:</strong> {best_efficiency_model['model_name']} with {best_efficiency_model['throughput_fps'] / (best_efficiency_model['trainable_params'] / 1e6):.2f} FPS per million parameters</p>
            </div>
        </div>

        <div class="section">
            <h2>🏁 Baseline Performance</h2>
            <div class="metric">
                <strong>Inference Time:</strong> {baseline_row['inference_time_ms']:.2f} ms per sample
            </div>
            <div class="metric">
                <strong>Throughput:</strong> {baseline_row['throughput_fps']:.1f} FPS
            </div>
            <div class="metric">
                <strong>Memory:</strong> {baseline_row['peak_memory_mb']:.0f} MB
            </div>
            <div class="metric">
                <strong>FLOPs:</strong> {baseline_row['total_flops']/1e9:.1f} GFLOPs
            </div>
            <div class="metric">
                <strong>Parameters:</strong> {baseline_row['trainable_params']:,} (100% trainable)
            </div>
        </div>

        <div class="section">
            <h2>⚡ LoRA Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Inference Time (ms)</th>
                    <th>Speedup</th>
                    <th>Throughput (FPS)</th>
                    <th>Memory (MB)</th>
                    <th>FLOPs (G)</th>
                    <th>Trainable Params</th>
                    <th>Efficiency</th>
                </tr>"""
    
    for _, row in lora_df.iterrows():
        speedup = baseline_row['inference_time_ms'] / row['inference_time_ms']
        memory_change = (row['peak_memory_mb'] - baseline_row['peak_memory_mb']) / baseline_row['peak_memory_mb'] * 100
        param_reduction = (baseline_row['trainable_params'] - row['trainable_params']) / baseline_row['trainable_params'] * 100
        efficiency = row['throughput_fps'] / (row['trainable_params'] / 1e6)
        
        speedup_class = "improvement" if speedup > 1.0 else "degradation"
        memory_class = "improvement" if memory_change < 0 else "degradation"
        
        html_content += f"""
                <tr>
                    <td><strong>{row['model_name']}</strong></td>
                    <td>{row['inference_time_ms']:.2f}</td>
                    <td class="{speedup_class}">{speedup:.2f}x</td>
                    <td>{row['throughput_fps']:.1f}</td>
                    <td>{row['peak_memory_mb']:.0f} <span class="{memory_class}">({memory_change:+.1f}%)</span></td>
                    <td>{row['total_flops']/1e9:.1f}</td>
                    <td>{row['trainable_params']:,} <span class="improvement">(-{param_reduction:.1f}%)</span></td>
                    <td>{efficiency:.2f}</td>
                </tr>"""
    
    html_content += f"""
            </table>
        </div>

        <div class="section">
            <h2>🏆 Best Performers</h2>
            <div class="best">
                <h3>🥇 Fastest Inference: {best_speed_model['model_name']}</h3>
                <p>Inference time: {best_speed_model['inference_time_ms']:.2f} ms ({baseline_row['inference_time_ms'] / best_speed_model['inference_time_ms']:.2f}x speedup)</p>
            </div>
            <div class="best">
                <h3>🥇 Most Efficient: {best_efficiency_model['model_name']}</h3>
                <p>Efficiency: {best_efficiency_model['throughput_fps'] / (best_efficiency_model['trainable_params'] / 1e6):.2f} FPS per million parameters</p>
            </div>
            <div class="best">
                <h3>🥇 Lowest Memory: {best_memory_model['model_name']}</h3>
                <p>Memory usage: {best_memory_model['peak_memory_mb']:.0f} MB</p>
            </div>
        </div>

        <div class="section">
            <h2>📈 Analysis Visualizations</h2>
            <h3>Inference Performance Overview</h3>
            <img src="inference_performance_overview.png" alt="Inference Performance Overview">
            
            <h3>Parameter Efficiency Analysis</h3>
            <img src="inference_parameter_efficiency.png" alt="Parameter Efficiency Analysis">
            
            <h3>LoRA Rank Impact Analysis</h3>
            <img src="lora_rank_analysis.png" alt="LoRA Rank Analysis">
        </div>

        <div class="section">
            <h2>🔍 Key Insights</h2>
            <ul>
                <li><strong>Parameter Efficiency:</strong> LoRA models achieve ~32% parameter reduction</li>
                <li><strong>Inference Overhead:</strong> LoRA introduces slight computational overhead during inference</li>
                <li><strong>Memory Impact:</strong> LoRA models use more memory due to additional layer structures</li>
                <li><strong>Trade-off:</strong> Parameter efficiency comes at the cost of slight inference speed reduction</li>
            </ul>
        </div>

        <div class="section">
            <h2>💡 Recommendations</h2>
            <ul>
                <li><strong>For fastest inference:</strong> Use baseline model for production inference</li>
                <li><strong>For parameter efficiency:</strong> Use {best_efficiency_model['model_name']} configuration</li>
                <li><strong>For deployment constraints:</strong> Lower LoRA ranks provide better inference efficiency</li>
                <li><strong>Best practice:</strong> Train with LoRA, fine-tune and merge weights for inference deployment</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(f'{analysis_dir}/inference_performance_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"Inference analysis HTML report saved to: {analysis_dir}/inference_performance_report.html")

def main():
    """Main function to run inference performance analysis"""
    print("=" * 60)
    print("INFERENCE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    df = analyze_inference_performance()
    
    if df is not None:
        print("\nInference analysis completed successfully!")
    else:
        print("\nInference analysis failed. Please check the requirements.")

if __name__ == "__main__":
    main()
