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
    budget = [] if "-t" in extra else ["-t", "32"]
    cmd = [binary, "ani", "--pack", pack, "--ql", str(q), "-o", str(out)] + budget + extra
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    lines = out.read_text().splitlines()
    assert lines[0] == "query\tref\tani\taf\tc_ab\tc_ba"
    return collections.Counter(lines[1:])

for mode, opts in [("self", []), ("rect", ["--rl", str(r)])]:
    expected = run(main, mode + "-main", opts + ["--screen", "0"])
    exhaustive = run(new, mode + "-zero", opts + ["--screen", "0"])
    assert exhaustive == expected, (mode, "screen 0 differs from main")
    for label, budget in [("serial", ["-t", "1"]), ("two", ["-t", "2"]),
                          ("io2", ["-t", "8", "--io-threads", "2"]),
                          ("clamped", ["-t", "8", "--io-threads", "99"])]:
        # Avoid duplicate -t options in CLI11.
        # run() uses 32 unless an explicit budget is supplied.
        actual = run(new, mode + "-" + label, opts + ["--screen", "0"] + budget)
        assert actual == expected, (mode, label, "reader budget changed output")
    screened = run(new, mode + "-default", opts)
    default_expected = run(main, mode + "-main-default", opts)
    assert screened == default_expected, (mode, "default screen differs from main")
    assert not screened - exhaustive, "screen changed values or added pairs"
    assert collections.Counter({k: v for k, v in exhaustive.items() if float(k.split("\t")[2]) >= 90}) == collections.Counter({k: v for k, v in screened.items() if float(k.split("\t")[2]) >= 90})
# Repeated reference entries count once for document frequency. A cap of one
# keeps all markers when the references contain only one distinct genome.
cap_r = d / "cap-r"
cap_r.write_text((acc[0] + "\n") * 2)
capped = run(new, "cap", ["--rl", str(cap_r), "--screen", "100", "--max-marker-df", "1"])
assert sum(v for k, v in capped.items() if k.split("\t")[0] == k.split("\t")[1]) == 4
for flag, value in [("--screen", "-1"), ("--screen", "101"), ("--max-marker-df", "-1"), ("--io-threads", "-1"), ("-t", "0")]:
    result = subprocess.run([new, "ani", "--pack", pack, "--ql", str(q), flag, value], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode != 0, (flag, value)
print("ANI CLI regressions passed")
