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
@click.option('--cloud', default=30, type=int)
@click.option('--outdir', default='outputs')

def main(start, end, cloud, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
            # 1
    cmd1 = [sys.executable, 'scripts/01_stac_data_prep.py', '--start', start, '--end', end, '--cloud', str(cloud)]
    subprocess.check_call(cmd1)

    meta = f"data/raw/meta_AOI_{start}_{end}.json"

    # 2
    cmd2 = [sys.executable, 'scripts/02_compute_indices_and_terrain.py', '--meta', meta]
    subprocess.check_call(cmd2)

    stack = f"data/processed/stack_AOI_{start}_{end}.tif"
    out = f"{outdir}/onion_AOI_{start}_{end}.tif"

    # 3
    cmd3 = [sys.executable, 'scripts/03_threshold_and_export.py', '--stack', stack, '--out', out]
    subprocess.check_call(cmd3)

    print('Pipeline finished. Results in', outdir)

if __name__ == '__main__':
    main()