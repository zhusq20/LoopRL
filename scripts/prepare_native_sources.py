"""Fetch pinned native sources and apply slime's required patch stacks plus Ouro.

Run with Python 3.12+. Existing directories are verified, never overwritten.
The manifest records archive, patch and modified-source SHA-256 digests.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request


REVISIONS = {
    "Megatron-LM": ("NVIDIA/Megatron-LM", "ae76c60e6c35d913867d6dd7bad0f30b128b54e7",
                    ["megatron.patch", "megatron-sglang-aligned.patch"]),
    "sglang": ("sgl-project/sglang", "e00930c5489053f26d86b179cee0d087f846acbb",
               ["sglang.patch", "sglang-top_p.patch", "sglang-release_hicache.patch",
                "sglang-pull_weights.patch", "sglang-deterministic.patch",
                "sglang-radio_layerscale.patch", "sglang-nemotron-multi-image.patch"]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("/tmp/looprl-native-sources"))
    parser.add_argument("--slime-path", type=Path, default=root / "third_party/slime")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    patch_root = args.slime_path / "docker/patch/v0.5.21"
    for name, (repo, revision, patch_names) in REVISIONS.items():
        patches = [patch_root / item for item in patch_names]
        if name == "sglang":
            patches.append(root / "integrations/slime/native/sglang-ouro.patch")
        if name == "Megatron-LM":
            patches.append(root / "integrations/slime/native/megatron-adam-resume.patch")
        expected = {patch.name: sha(patch) for patch in patches}
        destination = args.destination / f"{name}-{revision}"
        manifest_path = destination / "looprl-source-manifest.json"
        if destination.exists():
            if not manifest_path.exists():
                raise ValueError(f"Unmanaged source directory {destination}; choose a new destination")
            manifest = json.loads(manifest_path.read_text())
            assert manifest["revision"] == revision and manifest["patches"] == expected
            for relative, digest in manifest["modified_sources"].items():
                assert sha(destination / relative) == digest, f"Modified source: {relative}"
            print(f"Verified {destination}")
            continue
        archive = args.destination / f"{name}-{revision}.tar.gz"
        url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as bundle:
            bundle.extractall(args.destination, filter="data")
        touched = set()
        for patch in patches:
            subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=destination, check=True)
            subprocess.run(["git", "apply", str(patch.resolve())], cwd=destination, check=True)
            touched.update(line[6:] for line in patch.read_text().splitlines() if line.startswith("+++ b/"))
        manifest = dict(repository=repo, revision=revision, archive_sha256=sha(archive), patches=expected,
                        modified_sources={path: sha(destination / path) for path in sorted(touched)})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {destination}")


if __name__ == "__main__":
    main()
