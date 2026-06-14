"""
重新运行修改过的非ML实验脚本
"""
import subprocess
import json
import sys
import re
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).parent
PROJECT_DIR = SRC_DIR.parent
OUTPUT_JSON = PROJECT_DIR / 'experiments' / 'fixed_experiment_results.json'

SCRIPTS = [
    {'name': 'run_week8_tutorial', 'path': SRC_DIR / 'run_week8_tutorial.py', 'timeout': 600},
    {'name': 'run_epu_weight_experiment', 'path': SRC_DIR / 'run_epu_weight_experiment.py', 'timeout': 600},
    {'name': 'run_turnover_epu_experiment', 'path': SRC_DIR / 'run_turnover_epu_experiment.py', 'timeout': 600},
    {'name': 'run_week10_comprehensive', 'path': SRC_DIR / 'run_week10_comprehensive.py', 'timeout': 600},
]

def run_script(script_info):
    name = script_info['name']
    path = script_info['path']
    timeout = script_info['timeout']

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr

        metrics = extract_metrics(stdout + stderr)

        print(f"{'✅' if result.returncode == 0 else '❌'} {name} 完成 ({elapsed:.1f}s)")
        if metrics:
            print(f"   指标: {metrics}")

        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'returncode': result.returncode,
            'elapsed_sec': round(elapsed, 2),
            'metrics': metrics,
            'stdout_last_2000': stdout[-2000:] if stdout else '',
            'stderr_last_1000': stderr[-1000:] if stderr else '',
        }
    except subprocess.TimeoutExpired:
        print(f"❌ {name} 超时")
        return {'status': 'timeout', 'error': f'Timeout after {timeout}s'}
    except Exception as e:
        print(f"❌ {name} 异常: {e}")
        return {'status': 'exception', 'error': str(e)}

def extract_metrics(text):
    metrics = {}
    patterns = {
        'annual_return': r'年化收益[率]?[:\s]*([\d.]+)%',
        'sharpe_ratio': r'夏普比率?[:\s]*([\d.]+)',
        'max_drawdown': r'最大回撤[:\s]*-?([\d.]+)%',
        'annual_turnover': r'年化换手率?[:\s]*([\d.]+)%',
        'volatility': r'年化波动[率]?[:\s]*([\d.]+)%',
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            try:
                metrics[key] = float(matches[-1])
            except ValueError:
                pass
    return metrics

def main():
    # 加载已有结果（保留run_experiments和run_optimization_experiments）
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {'run_time': datetime.now().isoformat(), 'scripts': {}}

    for script_info in SCRIPTS:
        result = run_script(script_info)
        data['scripts'][script_info['name']] = result
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已更新至: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
