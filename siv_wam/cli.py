"""Dependency-light project entry point; GPU imports happen in child processes."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    'train-base': ('siv_wam.train', 'robotwin_train'),
    'train-siv': ('siv_wam.train_siv', 'robotwin_mask_joint'),
    'serve': ('siv_wam.server', 'robotwin_b2'),
}


def build_command(command, *, gpus=1, arguments=()):
    if gpus < 1:
        raise ValueError('gpus must be at least 1')
    module, config = COMMANDS[command]
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={gpus}', '-m', module, '--config-name', config,
            *arguments]


def diagnose(runtime=False):
    errors = []
    files = sorted((ROOT / 'siv_wam').rglob('*.py'))
    for path in files:
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    result = {'project': 'SIV-WAM', 'python': sys.version.split()[0],
              'python_files_checked': len(files), 'errors': errors,
              'scope': 'Static syntax only; no model execution.'}
    if runtime:
        packages = ['torch', 'torchvision', 'diffusers', 'transformers', 'safetensors',
                    'easydict', 'einops', 'numpy', 'scipy', 'PIL', 'cv2', 'tqdm',
                    'wandb', 'websockets', 'msgpack', 'lerobot', 'av']
        missing = [p for p in packages if importlib.util.find_spec(p) is None]
        errors.extend(f'Missing module: {p}' for p in missing)
        assets = {name: bool(os.getenv(name)) and Path(os.environ[name]).exists()
                  for name in ('SIV_WAM_PRETRAINED_PATH', 'SIV_WAM_DATASET_PATH',
                               'SIV_WAM_TRANSFORMER_PATH')}
        errors.extend(f'Missing asset: {name}' for name, ok in assets.items() if not ok)
        result['assets'] = assets
        if 'torch' not in missing:
            try:
                import torch
                result['cuda_available'] = torch.cuda.is_available()
                if not result['cuda_available']:
                    errors.append('CUDA is unavailable')
            except Exception as exc:
                errors.append(f'PyTorch load failed: {exc}')
        result['scope'] = 'Environment preflight; checkpoint compatibility and execution remain unverified.'
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'train-stage2':
        from .train_stage2 import main as stage2_main
        return stage2_main(argv[1:])
    parser = argparse.ArgumentParser(prog='siv-wam', description='SIV-WAM training and inference toolkit')
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Check source syntax and optionally runtime prerequisites')
    doctor.add_argument('--runtime', action='store_true')
    sub.add_parser('train-stage2', help='Stage II GRPO; use train-stage2 --help for options')
    for name in COMMANDS:
        child = sub.add_parser(name, help=f'Launch {name} on a single GPU host')
        child.add_argument('--gpus', type=int, default=1)
        child.add_argument('--dry-run', action='store_true')
    args, rest = parser.parse_known_args(argv)
    if args.command == 'doctor':
        if rest:
            parser.error('Unexpected arguments: ' + ' '.join(rest))
        result = diagnose(args.runtime)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return int(bool(result['errors']))
    if args.gpus < 1:
        parser.error('--gpus must be at least 1')
    command = build_command(args.command, gpus=args.gpus, arguments=rest)
    if args.dry_run:
        print(json.dumps(command, indent=2, ensure_ascii=False))
        return 0
    return subprocess.call(command, cwd=ROOT)
