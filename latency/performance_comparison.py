#!/usr/bin/env python3
"""
Performance comparison between original and optimized training scripts
"""

import time
import sys
import os
import numpy as np
import subprocess
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_with_profiling(script_name, args_dict, duration_limit=300):
    """Run a script with profiling and return timing results"""
    print(f"\n🔥 Running {script_name}...")
    
    # Prepare command
    cmd = [sys.executable, script_name]
    for key, value in args_dict.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Run with timeout
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration_limit
        )
        end_time = time.time()
        
        return {
            'success': True,
            'duration': end_time - start_time,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'duration': duration_limit,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e)
        }

def compare_performance():
    """Compare performance between original and optimized versions"""
    
    print("训练脚本性能对比测试")
    print("=" * 60)
    
    # Test configuration
    test_configs = [
        {
            'name': 'Small Scale',
            'N': 50,
            'M': 2,
            'episodes': 3,
            'description': '小规模测试 (50节点, 2分区, 3轮)'
        },
        {
            'name': 'Medium Scale',
            'N': 100, 
            'M': 2,
            'episodes': 2,
            'description': '中等规模测试 (100节点, 2分区, 2轮)'
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n📊 {config['name']} - {config['description']}")
        print("-" * 50)
        
        # Common arguments
        args = {
            'N': config['N'],
            'M': config['M'],
            'episodes': config['episodes'],
            'K': 3,
            'bs': 16,
            'seed': 42
        }
        
        config_results = {}
        
        # Test original version
        if os.path.exists('train_parallel_improved.py'):
            print("🔴 测试原始版本...")
            original_result = run_with_profiling('train_parallel_improved.py', args)
            config_results['original'] = original_result
            
            if original_result['success']:
                print(f"   ✅ 完成时间: {original_result['duration']:.2f}秒")
            else:
                print(f"   ❌ 失败: {original_result.get('error', 'Unknown error')}")
        
        # Test optimized version
        if os.path.exists('train_parallel_optimized.py'):
            print("🟢 测试优化版本...")
            optimized_result = run_with_profiling('train_parallel_optimized.py', args)
            config_results['optimized'] = optimized_result
            
            if optimized_result['success']:
                print(f"   ✅ 完成时间: {optimized_result['duration']:.2f}秒")
            else:
                print(f"   ❌ 失败: {optimized_result.get('error', 'Unknown error')}")
        
        # Calculate improvement
        if ('original' in config_results and 'optimized' in config_results and 
            config_results['original']['success'] and config_results['optimized']['success']):
            
            original_time = config_results['original']['duration']
            optimized_time = config_results['optimized']['duration']
            improvement = original_time / optimized_time
            time_saved = original_time - optimized_time
            
            print(f"\n📈 性能提升:")
            print(f"   原始版本: {original_time:.2f}秒")
            print(f"   优化版本: {optimized_time:.2f}秒")
            print(f"   提升倍数: {improvement:.2f}x")
            print(f"   节省时间: {time_saved:.2f}秒 ({time_saved/original_time*100:.1f}%)")
            
            config_results['improvement'] = improvement
            config_results['time_saved'] = time_saved
        
        results[config['name']] = config_results
    
    # Summary report
    print("\n" + "=" * 60)
    print("📋 性能对比总结")
    print("=" * 60)
    
    total_improvements = []
    
    for config_name, config_results in results.items():
        print(f"\n🔍 {config_name}:")
        
        if 'improvement' in config_results:
            improvement = config_results['improvement']
            time_saved = config_results['time_saved']
            
            print(f"   ✅ 性能提升: {improvement:.2f}x")
            print(f"   ⏱️ 节省时间: {time_saved:.2f}秒")
            
            total_improvements.append(improvement)
        else:
            print("   ❌ 无法计算性能提升（测试失败）")
    
    if total_improvements:
        avg_improvement = np.mean(total_improvements)
        print(f"\n🏆 平均性能提升: {avg_improvement:.2f}x")
        
        # Extrapolate to full training
        if avg_improvement > 1:
            original_epoch_time = 60  # seconds
            optimized_epoch_time = original_epoch_time / avg_improvement
            full_training_episodes = 1000
            
            original_total_time = original_epoch_time * full_training_episodes / 60  # minutes
            optimized_total_time = optimized_epoch_time * full_training_episodes / 60  # minutes
            time_saved_hours = (original_total_time - optimized_total_time) / 60
            
            print(f"\n🚀 完整训练时间预估 (1000轮):")
            print(f"   原始版本: {original_total_time:.1f}分钟")
            print(f"   优化版本: {optimized_total_time:.1f}分钟")
            print(f"   节省时间: {time_saved_hours:.1f}小时")
    
    # Save results
    results_file = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 详细结果已保存到: {results_file}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 建议")
    print("=" * 60)
    
    print("1. 如果优化效果显著，请使用 train_parallel_optimized.py 进行完整训练")
    print("2. 监控GPU使用率，确保硬件资源充分利用")
    print("3. 考虑进一步优化：")
    print("   - 使用更大的批次大小 (batch_size)")
    print("   - 启用GPU加速")
    print("   - 调整同步频率 (sync_freq)")
    print("4. 定期运行性能分析以发现新的瓶颈")
    
    return results

def run_memory_comparison():
    """Compare memory usage between versions"""
    print("\n" + "=" * 60)
    print("🧠 内存使用对比")
    print("=" * 60)
    
    try:
        import psutil
        import threading
        import subprocess
        
        def monitor_memory(process, results, duration=30):
            """Monitor memory usage of a process"""
            max_memory = 0
            memory_samples = []
            
            for _ in range(duration):
                try:
                    if process.poll() is None:  # Process still running
                        memory_info = process.memory_info()
                        current_memory = memory_info.rss / 1024 / 1024  # MB
                        max_memory = max(max_memory, current_memory)
                        memory_samples.append(current_memory)
                    time.sleep(1)
                except:
                    break
            
            results['max_memory'] = max_memory
            results['avg_memory'] = np.mean(memory_samples) if memory_samples else 0
        
        # Test both versions
        versions = [
            ('train_parallel_improved.py', 'original'),
            ('train_parallel_optimized.py', 'optimized')
        ]
        
        memory_results = {}
        
        for script, name in versions:
            if os.path.exists(script):
                print(f"📊 监控 {name} 版本内存使用...")
                
                # Start process
                process = subprocess.Popen([
                    sys.executable, script,
                    '--N', '100', '--M', '2', '--episodes', '1'
                ])
                
                # Monitor memory in background
                results = {}
                monitor_thread = threading.Thread(
                    target=monitor_memory,
                    args=(process, results, 60)
                )
                monitor_thread.start()
                
                # Wait for completion
                try:
                    process.wait(timeout=120)
                    monitor_thread.join()
                    
                    memory_results[name] = results
                    print(f"   最大内存: {results.get('max_memory', 0):.1f} MB")
                    print(f"   平均内存: {results.get('avg_memory', 0):.1f} MB")
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    print(f"   超时终止")
        
        # Compare results
        if 'original' in memory_results and 'optimized' in memory_results:
            original_max = memory_results['original']['max_memory']
            optimized_max = memory_results['optimized']['max_memory']
            memory_reduction = (original_max - optimized_max) / original_max * 100
            
            print(f"\n📈 内存使用对比:")
            print(f"   原始版本峰值: {original_max:.1f} MB")
            print(f"   优化版本峰值: {optimized_max:.1f} MB")
            print(f"   内存节省: {memory_reduction:.1f}%")
    
    except ImportError:
        print("❌ 需要安装 psutil 来监控内存使用:")
        print("   pip install psutil")

if __name__ == '__main__':
    print("性能对比工具")
    print("=" * 60)
    
    # Check if files exist
    files_to_check = [
        'train_parallel_improved.py',
        'train_parallel_optimized.py',
        'env_parallel_optimized.py'
    ]
    
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        print("请确保所有优化文件都已创建")
        sys.exit(1)
    
    # Run performance comparison
    results = compare_performance()
    
    # Optional memory comparison
    print("\n" + "=" * 60)
    response = input("是否运行内存使用对比? (y/n): ")
    if response.lower() == 'y':
        run_memory_comparison()
    
    print("\n✅ 性能对比完成!")
    print("使用优化版本: python train_parallel_optimized.py") 