"""Two immutable observation sets, final-only diagnostics, and comparison indexes."""
import json
import math
from collections import Counter, defaultdict
from itertools import combinations, groupby
from pathlib import Path

import numpy as np
from core import atomic_json, summary, reprojections

DDL = '''
CREATE TABLE final_input(track TEXT,image INT,feature INT,x REAL,y REAL,component INT,assignment TEXT,PRIMARY KEY(track,image));
CREATE INDEX final_input_feature ON final_input(image,feature);
CREATE TABLE final_observations(component INT,track TEXT,image INT,feature INT,x REAL,y REAL,PRIMARY KEY(component,track,image));
CREATE INDEX final_feature ON final_observations(component,image,feature);
CREATE TABLE ambiguous_points(component INT,track TEXT,PRIMARY KEY(component,track));
CREATE TABLE track_changes(scope INT,track TEXT,raw_length INT,final_length INT,retained INT,loss INT,added INT,status TEXT,coordinate_changes INT,PRIMARY KEY(scope,track));
CREATE INDEX change_track_state ON track_changes(scope,status,loss);
CREATE TABLE image_changes(scope INT,image INT,raw_count INT,final_count INT,retained INT,loss INT,added INT,PRIMARY KEY(scope,image));
CREATE TABLE pair_changes(scope INT,a INT,b INT,raw_count INT,final_count INT,PRIMARY KEY(scope,a,b));
CREATE INDEX change_pair_reverse ON pair_changes(scope,b);
'''


def read_sources(sfm):
    from opensfm import pymap
    managers, sources = {}, {}
    for source, name in [('matched', 'tracks.csv'), ('final', 'tracks_reconstructed.csv')]:
        path = sfm / name
        status = dict(path=str(path), status='missing', tracks=None, observations=None, images=None)
        manager = None
        if path.exists():
            try:
                with path.open('rb') as stream:
                    header=stream.readline(80)
                    if header not in (b'OPENSFM_TRACKS_VERSION_v102\n',b'OPENSFM_TRACKS_VERSION_v2\n'):
                        raise ValueError('不支持或损坏的 Tracks 文件头')
                    if header.startswith(b'OPENSFM_TRACKS_VERSION_v102') and not stream.read(1):
                        raise ValueError('二进制 Tracks 数据被截断')
                manager = pymap.TracksManager.instanciate_from_file(str(path))
                status.update(status='ready', tracks=manager.num_tracks(), images=manager.num_shots(),
                              observations=sum(len(manager.get_shot_observations(s)) for s in manager.get_shot_ids()))
            except Exception as error:
                status.update(status='unreadable', message=str(error))
        managers[source], sources[source] = manager, status
    return managers, sources


def finite(value):
    return float(value) if np.isfinite(value) else None


def build_final(db, recs, ids, manager, sources, out, camera_factory, warnings):
    """Preserve legacy residuals as references, then replace all official aggregates."""
    from opensfm import pygeometry
    db.executescript(DDL)
    db.execute('ALTER TABLE errors RENAME TO reference_errors')
    db.executescript('''CREATE TABLE errors(component INT,track TEXT,image INT,error REAL,px REAL,py REAL,state TEXT,PRIMARY KEY(component,track,image));
        CREATE INDEX final_error_image ON errors(component,image,error);
        DELETE FROM image_stats; DELETE FROM track_stats; DELETE FROM point_stats;''')
    memberships = defaultdict(set)
    for c, r in enumerate(recs):
        for track in r.get('points', {}):
            memberships[track].add(c)
    shot_memberships = defaultdict(set)
    for c, image in db.execute('SELECT component,image FROM shots'):
        shot_memberships[image].add(c)
    issues = Counter()
    if manager is not None:
        for name in manager.get_shot_ids():
            image = ids[name]
            batch, assigned = [], []
            for track, obs in manager.get_shot_observations(name).items():
                candidates = memberships[track] & shot_memberships[image]
                c = next(iter(candidates)) if len(candidates) == 1 else None
                assignment = 'assigned' if c is not None else 'ambiguous' if candidates else 'unassigned'
                x, y = map(finite, obs.point)
                batch.append((track, image, obs.id, x, y, c, assignment))
                if c is not None:
                    assigned.append((c, track, image, obs.id, x, y))
                else:
                    issues[assignment] += 1
                    db.executemany('INSERT OR IGNORE INTO ambiguous_points VALUES(?,?)', [(k, track) for k in candidates])
            db.executemany('INSERT INTO final_input VALUES(?,?,?,?,?,?,?)', batch)
            db.executemany('INSERT INTO final_observations VALUES(?,?,?,?,?,?)', assigned)
    # Affected points cannot claim a complete final error statistic.
    for c, track in db.execute('SELECT component,track FROM ambiguous_points'):
        issues['ambiguous_points'] += 1
    sources['final']['integrity'] = dict(issues)
    if issues:
        warnings.append('最终 Tracks 归属异常: ' + json.dumps(dict(issues), ensure_ascii=False))
    results = []
    for c, r in enumerate(recs):
        for name, shot in r.get('shots', {}).items():
            image = ids[name]
            rows = db.execute('''SELECT o.track,o.x,o.y,p.x,p.y,p.z FROM final_observations o
                JOIN points p ON p.component=o.component AND p.track=o.track WHERE o.component=? AND o.image=?''', (c, image)).fetchall()
            try:
                cam = camera_factory(r['cameras'][shot['camera']])
            except ValueError:
                cam = None
            if rows and cam:
                values = reprojections(cam, pygeometry.Pose(shot['rotation'], shot['translation']),
                                       np.asarray([v[3:] for v in rows], dtype=float), np.asarray([v[1:3] for v in rows], dtype=float))
            else:
                values = [(None, None, None, 'unsupported_camera')] * len(rows)
            db.executemany('INSERT INTO errors VALUES(?,?,?,?,?,?,?)', [(c, row[0], image, *v) for row, v in zip(rows, values)])
            errors = [v[0] for v in values if v[0] is not None]
            st = summary(errors)
            db.execute('INSERT INTO image_stats VALUES(?,?,?,?,?,?,?)', (c, image, st['mean'], st['median'], st['p95'], st['max'], st['count']))
        ambiguous = {t for (t,) in db.execute('SELECT track FROM ambiguous_points WHERE component=?', (c,))}
        point_values = defaultdict(list)
        obs_values = []
        for track, error in db.execute('SELECT track,error FROM errors WHERE component=? AND state="valid"', (c,)):
            obs_values.append(error)
            if track not in ambiguous:
                point_values[track].append(error)
        means = []
        for idx, track in db.execute('SELECT idx,track FROM points WHERE component=? ORDER BY idx', (c,)).fetchall():
            st = summary(point_values[track])
            means.append(st['mean'])
            db.execute('INSERT INTO point_stats VALUES(?,?,?,?,?,?,?,?)', (c, idx, st['mean'], st['median'], st['p95'], st['max'], st['rmse'], st['count']))
            db.execute('INSERT INTO track_stats VALUES(?,?,?,?,?,?,?)', (c, track, st['mean'], st['median'], st['p95'], st['max'], st['count']))
        np.asarray([v if v is not None else np.nan for v in means], dtype='<f8').tofile(out / ('point-errors-%d.bin' % c))
        st = summary([v for v in means if v is not None])
        st.update(points=len(means), unavailable=len(means)-st['count'], above_threshold=sum(v is not None and v >= 3 for v in means),
                  metric='mean_per_point_px', source='final', source_status=sources['final']['status'], ambiguous_points=len(ambiguous),
                  observation_errors=summary(obs_values), states=dict(db.execute('SELECT state,count(*) FROM errors WHERE component=? GROUP BY state', (c,))))
        results.append(st)
    db.commit()
    return results


def scope_observations(db, source, scope):
    if source == 'matched':
        sql = 'SELECT o.track,o.image,o.feature,o.x,o.y FROM observations o'
        args = ()
        if scope >= 0:
            sql += ' JOIN shots s ON s.image=o.image AND s.component=?'
            args = (scope,)
    elif scope < 0:
        sql, args = 'SELECT track,image,feature,x,y FROM final_input', ()
    else:
        sql, args = 'SELECT track,image,feature,x,y FROM final_observations WHERE component=?', (scope,)
    return db.execute(sql + ' ORDER BY 1,2', args)


def build_comparison(db, recs, sources, out):
    for scope in [-1, *range(len(recs))]:
        raw_images, final_images, retained_images = Counter(), Counter(), Counter()
        raw_pairs, final_pairs = Counter(), Counter()
        raw_tracks, final_tracks = {}, {}
        available = sources['matched']['status'] == sources['final']['status'] == 'ready'
        ambiguous = db.execute('SELECT count(*) FROM ambiguous_points WHERE component=?', (scope,)).fetchone()[0]
        available = available and not ambiguous
        for source, image_counts, pairs, tracks in [('matched', raw_images, raw_pairs, raw_tracks), ('final', final_images, final_pairs, final_tracks)]:
            for track, group in groupby(scope_observations(db, source, scope), lambda row: row[0]):
                observations = {row[1]: tuple(row[2:]) for row in group}
                tracks[track] = observations
                image_counts.update(observations.keys())
                pairs.update(combinations(sorted(observations), 2))
        changes, states, migrations = [], Counter(), Counter()
        removed_surviving = removed_without_point = 0
        surviving = {r[0] for r in db.execute('SELECT track FROM points' + (' WHERE component=?' if scope >= 0 else ''), (scope,) if scope >= 0 else ())}
        coordinate_changes = added_observations = 0
        for track in sorted(raw_tracks.keys() | final_tracks.keys()):
            a, b = raw_tracks.get(track, {}), final_tracks.get(track, {})
            same = {image for image in a.keys() & b.keys() if a[image][0] == b[image][0]}
            changed = sum(any(x is None or y is None or abs(x-y) > 1e-7 for x, y in zip(a[i][1:], b[i][1:])) for i in same)
            for image in same:
                retained_images[image] += 1
            loss, added = len(a)-len(same), len(b)-len(same)
            state = ('added' if not a else 'absent' if not b else 'changed' if added or changed else 'unchanged' if loss == 0 else 'shortened') if available else 'unknown'
            states[state] += 1
            if a and available:
                migrations[len(a), len(b)] += 1
                if track in surviving:
                    removed_surviving += loss
                else:
                    removed_without_point += loss
            coordinate_changes += changed
            added_observations += added
            changes.append((scope, track, len(a), len(b) if sources['final']['status']=='ready' else None,
                            len(same) if available else None, loss if available else None, added if available else None, state, changed))
        db.executemany('INSERT INTO track_changes VALUES(?,?,?,?,?,?,?,?,?)', changes)
        image_ids = [i for (i,) in db.execute('SELECT id FROM images' if scope < 0 else 'SELECT image FROM shots WHERE component=?', () if scope < 0 else (scope,))]
        db.executemany('INSERT INTO image_changes VALUES(?,?,?,?,?,?,?)',
                       [(scope, i, raw_images[i], final_images[i] if sources['final']['status']=='ready' else None,
                         retained_images[i] if available else None, raw_images[i]-retained_images[i] if available else None,
                         final_images[i]-retained_images[i] if available else None) for i in image_ids])
        db.executemany('INSERT INTO pair_changes VALUES(?,?,?,?,?)',
                       [(scope, a, b, raw_pairs[a,b], final_pairs[a,b] if sources['final']['status']=='ready' else None) for a,b in sorted(raw_pairs.keys() | final_pairs.keys())])
        raw_obs, final_obs = sum(raw_images.values()), sum(final_images.values())
        retained = sum(retained_images.values())
        info = dict(scope='project' if scope < 0 else 'component', component=scope, available=available, sources=sources,
                    ambiguity=ambiguous, raw_tracks=len(raw_tracks), final_tracks=len(final_tracks) if sources['final']['status']=='ready' else None,
                    raw_observations=raw_obs, final_observations=final_obs if sources['final']['status']=='ready' else None,
                    retained_observations=retained if available else None, track_retention=len(raw_tracks.keys() & final_tracks.keys())/len(raw_tracks) if available and raw_tracks else None,
                    observation_retention=retained/raw_obs if available and raw_obs else None, states=dict(states),
                    removed_surviving=removed_surviving if available else None, removed_without_point=removed_without_point if available else None,
                    added_observations=added_observations if available else None, coordinate_changes=coordinate_changes,
                    migrations=[dict(raw=a, final=b, count=n) for (a,b),n in sorted(migrations.items())])
        lengths = defaultdict(lambda: [0,0,0,0])
        for _, track, a,b,kept,loss,added,state,changed in changes:
            if a:
                group = lengths[a]
                group[0] += 1; group[1] += bool(b); group[2] += a; group[3] += kept or 0
        info['lengths'] = [dict(length=k, tracks=v[0], surviving=v[1] if available else None, raw_observations=v[2], retained=v[3] if available else None) for k,v in sorted(lengths.items())]
        atomic_json(out / ('comparison-%d.json' % scope), info)
        db.commit()
