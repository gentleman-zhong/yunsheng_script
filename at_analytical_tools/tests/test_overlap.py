import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from indexer import DDL
from track_comparison import DDL as COMPARISON_DDL
from overlap import component_overlap


@pytest.fixture
def db():
    connection = sqlite3.connect(':memory:')
    connection.executescript(DDL+COMPARISON_DDL)
    for image, width, height in [(0, 100, 80), (1, 800, 1000), (2, 200, 200), (3, 100, 80), (4, 100, 80)]:
        connection.execute('INSERT INTO images(id,width,height) VALUES(?,?,?)', (image, width, height))
    yield connection
    connection.close()


def add_points(db, start, count, images, component=0, u=.25):
    for image in images:
        db.execute('INSERT OR IGNORE INTO shots VALUES(?,?)', (component, image))
    for idx in range(start, start + count):
        track = str(idx)
        db.execute('INSERT INTO points VALUES(?,?,?,?,?,?)', (component, idx, track, 0, 0, 5))
        for image in images:
            width, height = db.execute('SELECT width,height FROM images WHERE id=?', (image,)).fetchone()
            x = (u * width - width / 2) / max(width, height)
            y = (.25 * height - height / 2) / max(width, height)
            db.execute('INSERT INTO final_observations VALUES(?,?,?,?,?,?)', (component, track, image, idx, x, y))
            db.execute('INSERT INTO errors VALUES(?,?,?,?,?,?,?)', (component, track, image, 3, x, y, 'valid'))


def test_complete_overlap_different_dimensions_and_fixed_threshold(db):
    add_points(db, 0, 20, [0, 1])
    result = component_overlap(db, 0)
    assert result['mean'] == pytest.approx(1)
    assert result['evaluated_images'] == result['registered_images'] == 2
    assert result['eligible_pairs'] == 1
    assert result['parameters']['max_error_px'] == 3


def test_partial_overlap_is_bidirectionally_averaged(db):
    add_points(db, 0, 20, [0, 1])
    add_points(db, 20, 20, [0])
    assert component_overlap(db, 0)['mean'] == pytest.approx(.75)
    # Swapping image identities preserves the score.
    for table in ['final_observations', 'errors', 'shots']:
        db.execute('UPDATE '+table+' SET image=image+10')
        db.execute('UPDATE '+table+' SET image=11-image')
    db.execute('UPDATE images SET id=id+10 WHERE id<2')
    db.execute('UPDATE images SET id=11-id WHERE id>=10')
    assert component_overlap(db, 0)['mean'] == pytest.approx(.75)


def test_dense_region_does_not_dominate_other_region(db):
    add_points(db, 0, 20, [0, 1])
    add_points(db, 20, 180, [0], u=.75)
    # A has two occupied cells, only one shared; B shares its entire observed region.
    assert component_overlap(db, 0)['mean'] == pytest.approx(.75)


def test_top_three_then_equal_image_weights(db):
    start = 0
    counts = [20, 30, 40, 50]
    for image, count in enumerate(counts, 1):
        add_points(db, start, count, [0, image])
        start += count
    scores = [(1 + count / sum(counts)) / 2 for count in counts]
    expected = (sum(sorted(scores)[-3:]) / 3 + sum(scores)) / 5
    result = component_overlap(db, 0)
    assert result['mean'] == pytest.approx(expected)
    assert result['eligible_pairs'] == 4
    assert result['evaluated_images'] == 5


@pytest.mark.parametrize('mutation', [
    "UPDATE errors SET error=3.001 WHERE image=1 AND track='0'",
    "UPDATE errors SET state='behind_camera' WHERE image=1 AND track='0'",
    "UPDATE errors SET state='unsupported_camera' WHERE image=1",
    "UPDATE final_observations SET x=100 WHERE image=1 AND track='0'",
    "UPDATE final_observations SET x=NULL WHERE image=1 AND track='0'",
    "UPDATE images SET width=0 WHERE id=1",
    "DELETE FROM points WHERE track='0'",
    "DELETE FROM shots WHERE image=1",
])
def test_invalid_or_insufficient_evidence_is_not_zero(db, mutation):
    add_points(db, 0, 20, [0, 1])
    db.execute(mutation)
    result = component_overlap(db, 0)
    assert result['mean'] is None
    assert result['evaluated_images'] == result['eligible_pairs'] == 0


def test_component_isolation_and_images_without_neighbors(db):
    add_points(db, 0, 20, [0, 1])
    db.execute('INSERT INTO shots VALUES(0,2)')
    # Reuse the same tracks in another component, with only one registered view.
    db.execute('INSERT INTO shots VALUES(1,0)')
    db.execute('INSERT INTO points SELECT 1,idx,track,x,y,z FROM points WHERE component=0')
    db.execute('INSERT INTO errors SELECT 1,track,image,error,px,py,state FROM errors WHERE component=0 AND image=0')
    first, second = component_overlap(db, 0), component_overlap(db, 1)
    assert first['mean'] == pytest.approx(1)
    assert (first['evaluated_images'], first['registered_images']) == (2, 3)
    assert second['mean'] is None and second['registered_images'] == 1
    assert component_overlap(db, 99)['mean'] is None
