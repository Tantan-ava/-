"""
批量运行所有修改过的实验脚本，收集修复后的性能指标。
由于完整运行可能耗时较长，此脚本会顺序执行并记录结果。
"""
import subprocess
import json
import os
import sys
import re
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SRC_DIR = PROJECT_DIR / 'src'
RESULTS_DIR = PROJECT_DIR / 'results'
EXPERIMENTS_DIR = PROJECT_DIR / 'experiments'
OUTPUT_JSON = EXPERIMENTS_DIR / 'fixed_experiment_results.json'

# 需要运行的脚本列表（按依赖关系和重要性排序）
SCRIPTS = [
    # 核心基线实验
    {'name': 'run_experiments', 'path': SRC_DIR / 'run_experiments.py', 'timeout': 600},
    # 优化实验
    {'name': 'run_optimization_experiments', 'path': SRC_DIR / 'run_optimization_experiments.py', 'timeout': 600},
    # ML实验
    {'name': 'run_week7_experiments', 'path': SRC_DIR / 'run_week7_experiments.py', 'timeout': 600},
    {'name': 'run_week7_ml_and_combo', 'path': SRC_DIR / 'run_week7_ml_and_combo.py', 'timeout': 600},
    # 价值因子
    {'name': 'run_week8_tutorial', 'path': SRC_DIR / 'run_week8_tutorial.py', 'timeout': 600},
    # EPU实验
    {'name': 'run_epu_weight_experiment', 'path': SRC_DIR / 'run_epu_weight_experiment.py', 'timeout': 600},
    {'name': 'run_turnover_epu_experiment', 'path': SRC_DIR / 'run_turnover_epu_experiment.py', 'timeout': 600},
    # 综合实验
    {'name': 'run_week10_comprehensive', 'path': SRC_DIR / 'run_week10_comprehensive.py', 'timeout': 600},
]

def run_script(script_info):
    """运行单个脚本并捕获输出"""
    name = script_info['name']
    path = script_info['path']
    timeout = script_info['timeout']

    if not path.exists():
        return {'status': 'missing', 'error': f'File not found: {path}'}

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(SRC_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr

        # 提取关键指标
        metrics = extract_metrics(stdout + stderr)

        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'returncode': result.returncode,
            'elapsed_sec': round(elapsed, 2),
            'metrics': metrics,
            'stdout_last_2000': stdout[-2000:] if stdout else '',
            'stderr_last_1000': stderr[-1000:] if stderr else '',
        }
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'error': f'Timeout after {timeout}s'}
    except Exception as e:
        return {'status': 'exception', 'error': str(e)}

def extract_metrics(text):
    """从输出文本中提取关键性能指标"""
    metrics = {}

    # 通用模式匹配
    patterns = {
        'annual_return': r'年化收益[率]?[:\s]*([\d.]+)%',
        'sharpe_ratio': r'夏普比率?[:\s]*([\d.]+)',
        'max_drawdown': r'最大回撤[:\s]*-?([\d.]+)%',
        'annual_turnover': r'年化换手率?[:\s]*([\d.]+)%',
        'total_return': r'累计收益[率]?[:\s]*([\d.]+)%',
        'volatility': r'年化波动[率]?[:\s]*([\d.]+)%',
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            # 取最后一个匹配（通常是最终结果）
            try:
                metrics[key] = float(matches[-1])
            except ValueError:
                pass

    return metrics

def main():
    all_results = {}
    overall_start = time.time()

    print(f"批量实验运行开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"预计运行时间: 较长（每个脚本最多10分钟）")

    for script_info in SCRIPTS:
        result = run_script(script_info)
        all_results[script_info['name']] = result

        # 每完成一个就保存一次（防止后续失败丢失前面结果）
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump({
                'run_time': datetime.now().isoformat(),
                'scripts': all_results,
            }, f, ensure_ascii=False, indent=2)

        if result['status'] == 'success':
            print(f"✅ {script_info['name']} 完成 ({result.get('elapsed_sec', 0)}s)")
            if result.get('metrics'):
                print(f"   提取指标: {result['metrics']}")
        else:
            print(f"❌ {script_info['name']} 失败: {result.get('error', result.get('status'))}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"全部完成! 总耗时: {overall_elapsed/60:.1f} 分钟")
    print(f"结果保存至: {OUTPUT_JSON}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
