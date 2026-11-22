"""The released figure script must produce both figures from the released CSVs alone,
without the optional panel-alignment module (it is not part of the package)."""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / 'results' / 'figures'


def test_make_figures_runs_without_layout_gate():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ('make_figures.py', 'deltaA.csv', 'expB.csv'):
            shutil.copy(FIG_DIR / name, Path(tmp) / name)
        env = dict(os.environ, MPLBACKEND='Agg')
        res = subprocess.run([sys.executable, 'make_figures.py'], cwd=tmp, env=env,
                             capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stderr[-2000:]
        assert 'panel-alignment check skipped' in res.stdout, 'the optional gate should be skipped in a clean directory'
        for name in ('fig_expA_v2.pdf', 'fig_expB_v2.pdf', 'fig_expA_v2.png', 'fig_expB_v2.png'):
            p = Path(tmp) / name
            assert p.exists() and p.stat().st_size > 1000, f'{name} missing or empty'
