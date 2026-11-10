"""Publish a trained stop-line model to the Hugging Face Hub (ernestchu/diorama), where the app fetches the latest.

Exports the checkpoint (export.py: Core ML and its ONNX twin), zips the .mlpackage, and uploads, in one commit:

    manifest.json                  version (one more than the Hub's), time, checkpoint, note, metrics, files' sha256
    StoplinePilot.mlpackage.zip    the Core ML model, for the app
    stopline_pilot.onnx            its ONNX twin, for the Linux development path (dioramaweb)

and tags the commit v<version>. The app compares the Hub's manifest.json with the one it has and, when the version is
newer, downloads the zip, checks its sha256, compiles it and switches over. Uses the `hf` login on this machine.

  python publish.py runs/fast/fast-s0/last.pt --note "stop line only" [--metrics metrics.json] [--dry-run]
  python publish.py EXPORT_DIR --checkpoint-name runs/fast/fast-s0 --version 0 --note "..."
      (an export.py output folder as it is: StoplinePilot.mlpackage and stopline_pilot.onnx)
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile

REPO = 'ernestchu/diorama'
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def zip_dir(src, dst):
  """A zip of the directory `src` with its own name at the top (StoplinePilot.mlpackage/...), entries sorted."""
  base = os.path.dirname(src)
  with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(src):
      dirs.sort()
      for name in sorted(files):
        p = os.path.join(root, name)
        z.write(p, os.path.relpath(p, base))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('checkpoint', help='a checkpoint to export, or a folder export.py already wrote')
  ap.add_argument('--checkpoint-name', help='what to record as the checkpoint (default: the argument)')
  ap.add_argument('--version', type=int, help='the version to publish (default: one more than the Hub\'s)')
  ap.add_argument('--note', required=True, help='what changed, for the manifest and the commit message')
  ap.add_argument('--metrics', help='a JSON file of test metrics to put in the manifest')
  ap.add_argument('--dry-run', action='store_true', help='export and write the folder, but upload nothing')
  args = ap.parse_args()

  from huggingface_hub import HfApi, hf_hub_download
  from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
  api = HfApi()
  version = args.version
  if version is None:
    try:
      version = json.load(open(hf_hub_download(REPO, 'manifest.json', force_download=True)))['version'] + 1
    except (EntryNotFoundError, RepositoryNotFoundError):
      version = 1

  out = tempfile.mkdtemp(prefix='diorama-publish-')
  if os.path.isdir(args.checkpoint):
    for n in ('StoplinePilot.mlpackage', 'stopline_pilot.onnx'):
      src, dst = os.path.join(args.checkpoint, n), os.path.join(out, n)
      (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)
  else:
    subprocess.run([sys.executable, os.path.join(HERE, 'export.py'), args.checkpoint, out], check=True)
  pkg = os.path.join(out, 'StoplinePilot.mlpackage')
  zip_dir(pkg, pkg + '.zip')
  shutil.rmtree(pkg)
  files = {n: {'sha256': sha256(os.path.join(out, n)), 'bytes': os.path.getsize(os.path.join(out, n))}
           for n in ('StoplinePilot.mlpackage.zip', 'stopline_pilot.onnx')}
  manifest = {
    'version': version,
    'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'checkpoint': args.checkpoint_name or args.checkpoint,
    'note': args.note,
    'metrics': json.load(open(args.metrics)) if args.metrics else None,
    'files': files,
  }
  json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
  print(json.dumps(manifest, indent=1))
  if args.dry_run:
    print('dry run: not uploaded;', out)
    return
  api.create_repo(REPO, repo_type='model', exist_ok=True)
  info = api.upload_folder(repo_id=REPO, folder_path=out, commit_message=f'v{version}: {args.note}')
  api.create_tag(REPO, tag=f'v{version}', revision=info.oid, tag_message=args.note)
  print(f'published v{version}: https://huggingface.co/{REPO} ({info.oid})')
  shutil.rmtree(out)


if __name__ == '__main__':
  main()
