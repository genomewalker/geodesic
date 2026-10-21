"""ANI CLI regression on a supplied pack; run on a compute node.

Usage: python3 tests/ani_cli_regression.py NEW MAIN PACK ACCESSIONS OUTPUT_DIR
OUTPUT_DIR must be a fresh directory. Requires only Python's standard library.
"""
import collections
import pathlib
import subprocess
import sys

new, main, pack, accessions, output = sys.argv[1:]
d = pathlib.Path(output)
d.mkdir()
acc = pathlib.Path(accessions).read_text().splitlines()[:4]
q = d / "q"
r = d / "r"
q.write_text("\n".join(acc + [acc[0], "__missing_ani_regression__"]) + "\n")
r.write_text("\n".join(reversed(acc + [acc[0]])) + "\n")

def run(binary, name, extra):
    out = d / (name + ".tsv")
    cmd = [binary, "ani", "--pack", pack, "--ql", str(q), "-t", "32", "-o", str(out)] + extra
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    lines = out.read_text().splitlines()
    assert lines[0] == "query\tref\tani\taf\tc_ab\tc_ba"
    return collections.Counter(lines[1:])

for mode, opts in [("self", []), ("rect", ["--rl", str(r)])]:
    expected = run(main, mode + "-main", opts)
    exhaustive = run(new, mode + "-zero", opts + ["--screen", "0"])
    assert exhaustive == expected, (mode, "screen 0 differs from main")
    screened = run(new, mode + "-default", opts)
    assert not screened - exhaustive, "screen changed values or added pairs"
    assert collections.Counter({k: v for k, v in exhaustive.items() if float(k.split("\t")[2]) >= 90}) == collections.Counter({k: v for k, v in screened.items() if float(k.split("\t")[2]) >= 90})
# Repeated reference entries count once for document frequency. A cap of one
# keeps genome-specific markers and therefore exact query/ref self matches.
capped = run(new, "cap", ["--rl", str(r), "--screen", "100", "--max-marker-df", "1"])
assert sum(v for k, v in capped.items() if k.split("\t")[0] == k.split("\t")[1]) == 7
for flag, value in [("--screen", "-1"), ("--screen", "101"), ("--max-marker-df", "-1")]:
    result = subprocess.run([new, "ani", "--pack", pack, "--ql", str(q), flag, value], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode != 0, (flag, value)
print("ANI CLI regressions passed")
