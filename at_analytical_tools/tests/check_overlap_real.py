"""Independently verify overlap using pairwise set intersections (read-only input)."""
import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from core import fingerprint
from server import create_app


def check(folder):
    meta = json.loads((folder / 'summary.json').read_text())
    assert fingerprint(meta['project'], meta['reconstruction']) == meta['fingerprint']
    db = sqlite3.connect('file:' + str(folder / 'index.sqlite') + '?mode=ro', uri=True)
    reports = []
    for result in meta['stats']['overlap']:
        c = result['component']
        params = result['parameters']
        grid = params['grid_size']
        images = {row[0]: defaultdict(set) for row in db.execute('SELECT image FROM shots WHERE component=?', (c,))}
        query = '''SELECT o.track,o.image,o.x,o.y,i.width,i.height,e.error,e.state
            FROM final_observations o JOIN images i ON i.id=o.image
            JOIN points p ON p.track=o.track AND p.component=o.component AND p.component=?
            LEFT JOIN errors e ON e.track=o.track AND e.image=o.image AND e.component=?'''
        for track, image, x, y, w, h, error, state in db.execute(query, (c, c)):
            if image not in images or state != 'valid' or error is None or not 0 <= error <= params['max_error_px']:
                continue
            if not w or not h or w <= 0 or h <= 0 or x is None or y is None:
                continue
            u, v = .5 + x * max(w, h) / w, .5 + y * max(w, h) / h
            if 0 <= u < 1 and 0 <= v < 1:
                images[image][(int(u * grid), int(v * grid))].add(track)
        points = {image: set().union(*cells.values()) for image, cells in images.items()}
        scores = defaultdict(list)
        pair_count = 0
        for a, b in combinations(sorted(images), 2):
            shared = points[a] & points[b]
            if len(shared) < params['min_common_tracks']:
                continue
            directional = [math.fsum(len(cell & shared) / len(cell) for cell in images[i].values()) / len(images[i]) for i in (a, b)]
            score = sum(directional) / 2
            scores[a].append(score)
            scores[b].append(score)
            pair_count += 1
        averages = [math.fsum(sorted(values, reverse=True)[:params['max_neighbors']]) / min(len(values), params['max_neighbors']) for values in scores.values()]
        expected = math.fsum(averages) / len(averages) if averages else None
        assert expected is None and result['mean'] is None or expected is not None and math.isclose(expected, result['mean'], abs_tol=1e-12)
        assert len(scores) == result['evaluated_images']
        assert len(images) == result['registered_images']
        assert pair_count == result['eligible_pairs']
        reports.append(dict(component=c, mean=expected, display=f'{expected*100:.1f}%' if expected is not None else '—',
                            registered_images=len(images), evaluated_images=len(scores), eligible_pairs=pair_count))
    db.close()
    store = SimpleNamespace(active=folder, status=lambda: {'available': True})
    response = create_app(store).test_client().get('/api/project?generation=' + folder.name)
    assert response.status_code == 200 and response.json['stats']['overlap'] == meta['stats']['overlap']
    assert fingerprint(meta['project'], meta['reconstruction']) == meta['fingerprint']
    return dict(source_unchanged=True, api_verified=True, independent_set_intersections=True, results=reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.index.resolve()), ensure_ascii=False, indent=2))
