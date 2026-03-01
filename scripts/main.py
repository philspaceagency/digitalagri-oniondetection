"""main.py

        A simple runner that executes the three scripts in order. It requires the user to input the start/end dates and province.
        """
import subprocess
import sys
import click
from pathlib import Path

@click.command()
@click.option('--start', required=True, help='Start date YYYY-MM-DD')
@click.option('--end', required=True, help='End date YYYY-MM-DD')
@click.option('--cloud', default=20, help='Max cloud cover for S2 search')
@click.option('--outdir', default='outputs', help='Directory to write final outputs')
def main(start, end, cloud, outdir):
    """Run the full pipeline: 01 -> 02 -> 03 using the AOI-based STAC search.

    This wrapper calls `scripts/01_stac_data_prep.py --run` which writes the
    meta file and invokes `02_compute_indices_and_terrain.py`. After that
    this script runs `03_threshold_and_export.py` on the produced stack.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # 1) create meta (AOI) and run step 02
    cmd1 = [
        sys.executable,
        'scripts/01_stac_data_prep.py',
        '--start', start,
        '--end', end,
        '--cloud', str(cloud),
        '--run',
    ]
    print('▶ Running 01_stac_data_prep.py -- this will generate meta and run step 02')
    subprocess.check_call(cmd1)

    # meta filename created by 01_stac_data_prep.py
    meta = f"data/raw/meta_AOI_{start}_{end}.json"

    # expected stack written by step 02
    stack = f"data/processed/stack_AOI_{start}_{end}.tif"
    out = f"{outdir}/onion_AOI_{start}_{end}.tif"

    # 3) threshold and export
    cmd3 = [sys.executable, 'scripts/03_threshold_and_export.py', '--stack', stack, '--out', out]
    print('▶ Running 03_threshold_and_export.py')
    subprocess.check_call(cmd3)

    print('Pipeline finished. Results in', outdir)


if __name__ == '__main__':
    main()