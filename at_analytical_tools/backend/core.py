"""Read-only project adapters. No algorithm stage or project writer is invoked."""
import gzip
import hashlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 4
ROOT = Path(__file__).resolve().parent.parent


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(str(tmp), str(path))


def project_paths(project, reconstruction=None):
    project = Path(project).resolve()
    sfm = project if project.name == 'opensfm' else project / 'opensfm'
    project = sfm.parent
    if not sfm.is_dir():
        raise ValueError('项目中没有 opensfm 目录')
    if reconstruction:
        rec = Path(reconstruction).resolve()
        if not rec.is_file():
            raise ValueError('指定的重建文件不存在')
    else:
        rec = sfm / 'reconstruction.topocentric.json'
        if not rec.is_file():
            rec = sfm / 'reconstruction.json'
    return project, sfm, rec


def image_paths(project, sfm):
    listing = sfm / 'image_list.txt'
    result = {}
    if listing.exists():
        for line in listing.read_text().splitlines():
            if not line.strip():
                continue
            raw = Path(line.strip())
            name = str(raw).split('images/', 1)[-1]
            candidate = project / 'images' / name
            if not candidate.is_file():
                candidate = raw if raw.is_absolute() else sfm / raw
            result[name] = str(candidate.resolve())
    else:
        for path in sorted((project / 'images').rglob('*')):
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
                result[str(path.relative_to(project / 'images'))] = str(path.resolve())
    return result


def feature_path(sfm, name):
    normalized = name.replace('/', '__')
    candidates = [sfm / 'features' / (normalized + '.features.npz'),
                  sfm / 'features' / (name + '.features.npz'),
                  sfm / 'features' / (normalized + '.npz')]
    return next((p for p in candidates if p.is_file()), candidates[0])


def load_points(path):
    with np.load(path, allow_pickle=False) as data:
        points = data['points']
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError('特征 points 不是 N×2 或 N×4 数组')
        return np.asarray(points[:, :2], dtype=np.float64)


class MatchingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in {
            ('numpy.core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
            ('numpy._core.multiarray', '_reconstruct'), ('numpy._core.multiarray', 'scalar'),
            ('numpy', 'ndarray'), ('numpy', 'dtype'),
        }:
            raise pickle.UnpicklingError('不支持的 pickle 类型: %s.%s' % (module, name))
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        return super().find_class(module, name)


def read_matches(path):
    with gzip.open(path, 'rb') as stream:
        value = MatchingUnpickler(stream).load()
    if not isinstance(value, dict):
        raise ValueError('匹配文件不是字典')
    return value


def canonical_pair(a, b, pairs):
    pairs = np.asarray(pairs)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('匹配索引必须为 N×2 整数数组')
    if pairs.size and (np.any(pairs > np.iinfo(np.int32).max) or np.any(pairs < np.iinfo(np.int32).min)):
        raise ValueError('匹配索引超出 int32 范围')
    if a > b:
        a, b, pairs = b, a, pairs[:, ::-1]
    return a, b, np.asarray(pairs, dtype=np.int32)


def pixel_coordinates(xy, width, height):
    return np.asarray(xy) * max(width, height) + np.array([width / 2 - .5, height / 2 - .5])


def reprojections(camera, pose, xyz, observed):
    """Unweighted Euclidean errors at original image resolution."""
    xyz = pose.transform_many(np.asarray(xyz, dtype=np.float64))
    with np.errstate(all='ignore'):
        predicted = camera.project_many(xyz)
        errors = np.linalg.norm(predicted - observed, axis=1) * max(camera.width, camera.height)
    result = []
    for p, e, z in zip(predicted, errors, xyz[:, 2]):
        state = 'behind_camera' if z <= 0 else ('invalid_projection' if not np.isfinite(e) or not np.isfinite(p).all() else 'valid')
        result.append((float(e) if state == 'valid' else None,
                       float(p[0]) if np.isfinite(p[0]) else None,
                       float(p[1]) if np.isfinite(p[1]) else None, state))
    return result


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0, mean=None, median=None, min=None, max=None, p90=None, p95=None, rmse=None)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()),
                max=float(a.max()), p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                rmse=float(np.sqrt(np.mean(a*a))))


def fingerprint(project, reconstruction=None):
    project, sfm, rec = project_paths(project, reconstruction)
    paths = [rec, sfm / 'tracks.csv', sfm / 'tracks_reconstructed.csv', sfm / 'image_list.txt', sfm / 'config.yaml']
    for folder in ['features', 'matches', 'exif']:
        paths.extend((sfm / folder).rglob('*'))
    paths.extend(Path(p) for p in image_paths(project, sfm).values())
    digest = hashlib.sha256(str(SCHEMA_VERSION).encode())
    final_tracks = sfm / 'tracks_reconstructed.csv'
    digest.update(b'final-present' if final_tracks.is_file() else b'final-missing')
    if final_tracks.is_file():
        with final_tracks.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    for path in sorted(set(paths)):
        if path.is_file():
            stat = path.stat()
            digest.update(('%s:%d:%d\n' % (path, stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()
