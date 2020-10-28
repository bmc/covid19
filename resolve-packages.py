# Run this from within the Jupyter environment (e.g., from a
# Jupyter Labs terminal) to create a usable list of packages.
# "pip freeze" in that environment shows "@ file:///" entries
# for many packages, which isn't helpful.
#
# python resolve-packages.py >requirements.txt

import subprocess
import re
from typing import Sequence

def run_command(cmd: Sequence[str]) -> Sequence[str]:
    return subprocess.check_output(cmd).decode('utf-8').split('\n')

ws = re.compile(r'\s+')
output = run_command(['pip', 'freeze'])
for line in output:
    if len(line.strip()) == 0:
        continue

    tokens = ws.split(line, maxsplit=1)
    if len(tokens) == 1:
        print(line)
        continue

    package = tokens[0]
    info_lines = run_command(['pip', 'show', package])
    for i in info_lines:
        if i.lower().startswith('version:'):
            version = ws.split(i)[1]
            print(f'{package}=={version}')
            break
    else:
        raise Exception(f'Cannot find version for {package}')
