"""Spatially balanced co-visibility, not a physical image-area measurement."""
import math
from collections import Counter, defaultdict
from itertools import combinations, groupby


GRID_SIZE = 16
MAX_NEIGHBORS = 3
MIN_COMMON_TRACKS = 20
MAX_ERROR_PX = 3.0


def component_overlap(db, component):
    registered = db.execute('SELECT count(*) FROM shots WHERE component=?', (component,)).fetchone()[0]
    query = '''SELECT o.track,o.image,o.x,o.y,i.width,i.height
        FROM points p JOIN final_observations o ON o.track=p.track AND o.component=p.component
        JOIN shots s ON s.component=p.component AND s.image=o.image
        JOIN images i ON i.id=o.image
        JOIN errors e ON e.component=p.component AND e.track=p.track AND e.image=o.image
        WHERE p.component=? AND NOT EXISTS(SELECT 1 FROM ambiguous_points ap WHERE ap.component=p.component AND ap.track=p.track) AND e.state='valid' AND e.error>=0 AND e.error<=?
        ORDER BY o.track,o.image'''

    def observations():
        for track, image, x, y, width, height in db.execute(query, (component, MAX_ERROR_PX)):
            if not width or not height or width <= 0 or height <= 0 or x is None or y is None:
                continue
            # Image boundaries are half a pixel outside the first/last pixel centers.
            u = (x * max(width, height) + width / 2) / width
            v = (y * max(width, height) + height / 2) / height
            if not (math.isfinite(u) and math.isfinite(v) and 0 <= u < 1 and 0 <= v < 1):
                continue
            yield track, image, int(v * GRID_SIZE) * GRID_SIZE + int(u * GRID_SIZE)

    cells = defaultdict(Counter)
    for _, image, cell in observations():
        cells[image][cell] += 1
    weights = {image: {cell: 1.0 / (len(counts) * count) for cell, count in counts.items()}
               for image, counts in cells.items()}

    # Only accumulate pairs that actually share points; never allocate an N x N matrix.
    pairs = {}
    for _, group in groupby(observations(), key=lambda row: row[0]):
        for (_, a, ca), (_, b, cb) in combinations(group, 2):
            pair = pairs.setdefault((a, b), [0, 0.0])
            pair[0] += 1
            pair[1] += (weights[a][ca] + weights[b][cb]) / 2

    neighbors = defaultdict(list)
    eligible_pairs = 0
    for (a, b), (count, score) in pairs.items():
        if count < MIN_COMMON_TRACKS:
            continue
        score = min(1.0, max(0.0, score))  # Bound floating-point roundoff.
        eligible_pairs += 1
        neighbors[a].append((score, b))
        neighbors[b].append((score, a))
    image_means = []
    for image in sorted(neighbors):
        top = sorted(neighbors[image], key=lambda item: (-item[0], item[1]))[:MAX_NEIGHBORS]
        image_means.append(math.fsum(score for score, _ in top) / len(top))
    return dict(component=component, metric='spatially_balanced_covisibility_final_v2', source='final',
                mean=math.fsum(image_means) / len(image_means) if image_means else None,
                registered_images=registered, evaluated_images=len(image_means),
                eligible_pairs=eligible_pairs,
                parameters=dict(grid_size=GRID_SIZE, max_neighbors=MAX_NEIGHBORS,
                                min_common_tracks=MIN_COMMON_TRACKS, max_error_px=MAX_ERROR_PX))
