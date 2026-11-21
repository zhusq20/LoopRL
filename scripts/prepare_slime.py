"""Fetch pinned slime and apply either native Ouro work or the HF reference patch."""
import argparse
from pathlib import Path
import subprocess

REVISION = "79aa41a884ab0bf65222c94418f6357d3c31192b"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destination", default=None)
    p.add_argument("--backend", choices=["native", "hf-reference"], default="native")
    args = p.parse_args()
    destination = Path(args.destination or ("third_party/slime" if args.backend == "native"
                                           else "third_party/slime-hf-reference")).resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/THUDM/slime.git", str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", REVISION], check=True)
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise RuntimeError("Existing checkout differs from the pinned revision; use a new destination")
    patch_root = Path(__file__).resolve().parents[1] / "integrations/slime"
    if args.backend == "native":
        reference_patch = patch_root / "custom-actor.patch"
        reference_present = subprocess.run(
            ["git", "-C", str(destination), "apply", "--reverse", "--check", str(reference_patch)],
            capture_output=True,
        ).returncode == 0
        if reference_present:
            raise RuntimeError("Native Ouro work needs a separate clean checkout; this checkout has the HF actor patch")
    elif (destination / "slime_plugins/models/ouro.py").exists():
        raise RuntimeError("HF reference work needs a separate checkout; this checkout has the native Ouro model")
    patch = patch_root / ("ouro-native.patch" if args.backend == "native" else "custom-actor.patch")
    reverse = subprocess.run(["git", "-C", str(destination), "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if reverse.returncode:
        subprocess.run(["git", "-C", str(destination), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(destination), "apply", str(patch)], check=True)
    print(f"slime {REVISION}: {args.backend} patch applied at {destination}")
    if args.backend == "native":
        print("Native Ouro integration is in progress; applying the patch is not training acceptance.")


if __name__ == "__main__":
    main()
