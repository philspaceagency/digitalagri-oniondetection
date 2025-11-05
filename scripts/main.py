"""main.py

        A simple runner that executes the three scripts in order. It requires the user to input the start/end dates and province.
        """
import subprocess
import sys
import click
from pathlib import Path

@click.command()
@click.option('--start', required=True)
@click.option('--end', required=True)
@click.option('--province', required=True)
@click.option('--gaul', default=None)
@click.option('--outdir', default='outputs')

def main(start, end, province, gaul, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
            # 1
    cmd1 = [sys.executable, 'scripts/01_date_prep_and_stac_download.py', '--start', start, '--end', end, '--province', province]
    if gaul:
        cmd1 += ['--gaul', gaul]
    subprocess.check_call(cmd1)

    meta = f"data/raw/meta_{province}_{start}_{end}.json"

    # 2
    cmd2 = [sys.executable, 'scripts/02_compute_indices_and_terrain.py', '--meta', meta]
    subprocess.check_call(cmd2)

    stack = f"data/processed/stack_{province}_{start}_{end}.tif"
    out = f"{outdir}/onion_{province}_{start}_{end}.tif"

    # 3
    cmd3 = [sys.executable, 'scripts/03_threshold_and_export.py', '--stack', stack, '--out', out]
    subprocess.check_call(cmd3)

    print('Pipeline finished. Results in', outdir)

if __name__ == '__main__':
    main()