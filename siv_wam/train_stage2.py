"""Run Stage II GRPO training/evaluation; help and config validation do not load CUDA."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from .rl.config import RLConfig


def read_config(path):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(spec) - {'backend', 'rl', 'model', 'environment'}:
        raise ValueError('Unknown top-level configuration keys')
    config = RLConfig(**spec.get('rl', {}))
    if spec.get('backend') not in {'smoke', 'robotwin'}:
        raise ValueError('backend must be smoke or robotwin')
    if spec['backend'] == 'robotwin':
        for key in ('root', 'task', 'task_config', 'instruction', 'model_quaternion_order', 'environment_quaternion_order'):
            if not spec.get('environment', {}).get(key):
                raise ValueError(f'Missing environment.{key}')
        for key in ('pretrained_path', 'transformer_path'):
            if not spec.get('model', {}).get(key):
                raise ValueError(f'Missing model.{key}')
    return spec, config


def make_backend(spec, config, output):
    if spec['backend'] == 'smoke':
        from .rl.smoke import SmokeActor, SmokeEnvironment
        return SmokeActor(config), SmokeEnvironment(), {'backend': 'smoke', 'implementation': 'gaussian_grpo_v1'}
    from .rl.wan_actor import WanStage2Actor
    from .rl.robotwin import RoboTwinEnvironment
    model = dict(spec['model'])
    environment = dict(spec['environment'])
    for key in ('pretrained_path', 'transformer_path'):
        model[key] = str(Path(os.path.expandvars(model[key])).expanduser().resolve())
        if not Path(model[key]).is_dir():
            raise FileNotFoundError(model[key])
    environment['root'] = str(Path(os.path.expandvars(environment['root'])).expanduser().resolve())
    checkpoint = Path(model['transformer_path'])
    fingerprint = [{'file': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                   for p in sorted(checkpoint.glob('*.safetensors'))]
    if not fingerprint:
        raise FileNotFoundError('Transformer checkpoint has no safetensors weights')
    config_hash = hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest()
    identity = {'backend': 'robotwin', 'model': model, 'environment': environment,
                'checkpoint_files': fingerprint, 'checkpoint_config_sha256': config_hash,
                'implementation': 'gaussian_grpo_v1'}
    actor = WanStage2Actor(model, config, output)
    env = RoboTwinEnvironment(environment)
    return actor, env, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description='SIV-WAM Stage II: grouped action-denoising GRPO')
    parser.add_argument('--config', required=True, help='JSON experiment configuration')
    parser.add_argument('--output', default='./outputs/stage2')
    parser.add_argument('--resume', help='Stage II latest.pt, including reference head and optimizer')
    parser.add_argument('--mode', choices=('train', 'evaluate'), default='train')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true', help='Evaluation: disable transition exploration noise')
    parser.add_argument('--check-config', action='store_true', help='Validate configuration without importing torch')
    args = parser.parse_args(argv)
    spec, config = read_config(args.config)
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if args.deterministic and args.mode != 'evaluate':
        parser.error('--deterministic is only allowed in evaluation')
    if args.check_config:
        print(json.dumps({'backend': spec['backend'], 'rl': asdict(config)}, indent=2))
        return 0
    output = Path(args.output).resolve()
    if args.mode == 'train' and not args.resume and (output / 'train.jsonl').exists():
        parser.error('Output already contains a run; use --resume or a different --output')
    import torch
    from .rl.trainer import GRPOTrainer
    torch.manual_seed(config.seed)
    actor, env, identity = make_backend(spec, config, output)
    try:
        trainer = GRPOTrainer(actor, env, config, output, identity)
        if args.resume:
            trainer.load(args.resume)
        (output / 'experiment.json').write_text(json.dumps({'config': spec, 'identity': identity,
            'torch': str(torch.__version__), 'config_hash': trainer.config_hash()}, indent=2), encoding='utf-8')
        if args.mode == 'train':
            trainer.train()
        else:
            result = trainer.evaluate(args.episodes, args.deterministic)
            print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
