"""Regression checks for relocation and the dependency-free command surface."""
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from siv_wam.cli import ROOT, COMMANDS, build_command, diagnose, main

SOURCE_FILES = [p for folder in ('siv_wam', 'scripts', 'evaluation', 'tests')
                for p in (ROOT / folder).rglob('*.py')]


class ProjectTests(unittest.TestCase):
    def test_all_python_sources_parse(self):
        for path in SOURCE_FILES:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding='utf-8'))

    def test_internal_imports_resolve_after_relocation(self):
        for path in SOURCE_FILES:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == 'siv_wam' or node.module.startswith('siv_wam.'):
                        target = ROOT.joinpath(*node.module.split('.'))
                        with self.subTest(path=path, module=node.module):
                            self.assertTrue(target.with_suffix('.py').exists() or (target / '__init__.py').exists())

    def test_commands_preserve_paths_with_spaces(self):
        args = ['--save-root', '/tmp/training output', '--resume-from', '/tmp/step 100']
        for command, (module, _) in COMMANDS.items():
            invocation = build_command(command, gpus=2, arguments=args)
            self.assertIn(module, invocation)
            self.assertEqual(invocation[-len(args):], args)
            self.assertIn('--nproc_per_node=2', invocation)
            self.assertTrue(ROOT.joinpath(*module.split('.')).with_suffix('.py').exists())

    def test_dry_run_never_launches_training(self):
        with patch('siv_wam.cli.subprocess.call') as launch, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['train-siv', '--dry-run', '--save-root', 'a b']), 0)
        launch.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())[-2:], ['--save-root', 'a b'])

    def test_invalid_gpu_count_rejected(self):
        with self.assertRaises(ValueError):
            build_command('serve', gpus=0)

    def test_help_does_not_load_gpu_dependencies(self):
        code = "import sys; from siv_wam.cli import main; main(['doctor']); assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['errors'], [])

    def test_runtime_reports_missing_assets(self):
        with patch.dict('os.environ', {}, clear=True), patch('siv_wam.cli.importlib.util.find_spec', return_value=None):
            report = diagnose(runtime=True)
        self.assertFalse(any(report['assets'].values()))
        self.assertTrue(report['errors'])


if __name__ == '__main__':
    unittest.main()
