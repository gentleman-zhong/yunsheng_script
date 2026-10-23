import json
import os
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from opensfm import pymap
from test_viewer import dataset
from core import fingerprint
from indexer import build
from server import create_app


@pytest.fixture
def project(dataset, tmp_path):
    project=tmp_path/'project'
    shutil.copytree(dataset[0],project)
    sfm=project/'opensfm'
    (sfm/'image_list.txt').write_text('\n'.join(str(project/'images'/name) for name in ['sub/a.jpg','sub/b.jpg','sub/c.jpg']))
    return project


def open_client(project,tmp_path):
    out=tmp_path/'new-index';build(project,out)
    store=SimpleNamespace(active=out,status=lambda:{'available':True})
    return create_app(store).test_client(),out


def final_subset(project, names=('sub/a.jpg','sub/b.jpg')):
    sfm=project/'opensfm'
    raw=pymap.TracksManager.instanciate_from_file(str(sfm/'tracks.csv'))
    final=raw.construct_sub_tracks_manager(['0'],list(names))
    final.write_to_file(str(sfm/'tracks_reconstructed.csv'))
    return raw,final


def test_partial_retention_final_errors_and_reference_are_separate(project,tmp_path):
    raw,final=final_subset(project)
    raw.remove_observation('sub/c.jpg','0')
    raw.add_observation('sub/c.jpg','0',pymap.Observation(10,10,.01,30,80,60,2))
    raw.write_to_file(str(project/'opensfm/tracks.csv'))
    client,out=open_client(project,tmp_path)
    summary=client.get('/api/comparison/summary?edge_threshold=1').json
    assert summary['states']=={'shortened':1,'absent':1}
    assert summary['raw_observations']==6 and summary['final_observations']==2
    assert summary['removed_surviving']==1 and summary['removed_without_point']==3
    assert summary['observation_retention']==pytest.approx(2/6)
    assert summary['topology']['matched']==dict(edges=3,components=1,isolated=0)
    assert summary['topology']['final']==dict(edges=1,components=2,isolated=1)
    assert summary['topology']['vanished']==2
    point=client.get('/api/scene/0/point/0').json
    assert point['stats']['count']==2 and point['stats']['mean']<1e-5
    assert len(point['observations'])==2 and len(point['reference_observations'])==1
    assert point['reference_observations'][0]['reference_error']>1000
    assert point['reference_observations'][0]['error'] is None
    feature=client.get('/api/images/2/features').json
    discarded=next(o for o in feature['observations'] if o['track']=='0')
    assert discarded['idx']==0 and discarded['retained'] is False
    assert discarded['fate']=='removed_observation'
    assert client.get('/api/images/2/neighbor-ids?relation=final').json==[]
    assert client.get('/api/comparison/tracks?state=shortened&raw_length=3&final_length=2').json['total']==1
    assert client.get('/api/comparison/pairs?edge_threshold=1&state=vanished').json['total']==2
    assert client.get('/api/comparison/pairs/0/2?state=lost').json['total']==2
    pair=client.get('/api/comparison/pairs/0/1?state=final').json
    reverse=client.get('/api/comparison/pairs/1/0?state=final').json
    assert pair['tracks']==['0'] and pair['retained']==[True]
    assert pair['a_points']==reverse['b_points'] and pair['pairs']==[p[::-1] for p in reverse['pairs']]
    assert client.get('/api/comparison/pairs/0/1?limit=1&offset=1').json['count']==1


def test_final_coordinates_are_used_and_high_residuals_not_cut(project,tmp_path):
    raw,final=final_subset(project)
    final.remove_observation('sub/a.jpg','0')
    final.add_observation('sub/a.jpg','0',pymap.Observation(.1,0,.01,30,80,60,0))
    final.write_to_file(str(project/'opensfm/tracks_reconstructed.csv'))
    client,out=open_client(project,tmp_path)
    point=client.get('/api/scene/0/point/0').json
    assert point['stats']['mean']==pytest.approx(5)
    assert point['stats']['count']==2
    assert client.get('/api/diagnostics?threshold=3').json['total']==1
    observations=client.get('/api/images/0/features').json['observations']
    kept=next(o for o in observations if o['retained'])
    assert kept['x']==pytest.approx(.1) and kept['coordinate_changed']
    assert kept['reference_error']<1e-5 and kept['error']==pytest.approx(10)
    assert client.get('/api/comparison/summary').json['states']['changed']==1
    pair=client.get('/api/comparison/pairs/0/1?state=final').json
    assert pair['a_points'][0][0]==pytest.approx(.1)


@pytest.mark.parametrize('state',['missing','unreadable','empty'])
def test_absent_final_does_not_imply_all_rejected(project,tmp_path,state):
    file=project/'opensfm/tracks_reconstructed.csv'
    if state=='missing':file.unlink()
    elif state=='unreadable':file.write_text('not a tracks file')
    else:file.write_text('OPENSFM_TRACKS_VERSION_v2\n')
    client,out=open_client(project,tmp_path)
    meta=client.get('/api/project').json['stats']
    assert meta['sources']['final']['status']==('ready' if state=='empty' else state)
    assert meta['errors'][0]['mean'] is None
    assert np.isnan(np.frombuffer(client.get('/api/scene/0/point-errors').data,dtype='<f8')).all()
    comparison=client.get('/api/comparison/summary').json
    assert comparison['available']==(state=='empty')
    if state!='empty':
        assert comparison['states']=={'unknown':2}
        assert comparison['removed_surviving'] is None
        assert comparison['final_observations'] is None
        assert all(o['fate']=='unknown' for o in client.get('/api/images/0/features').json['observations'])
    else:
        assert comparison['states']=={'absent':2}
        assert comparison['final_observations']==0


def test_missing_resources_preserve_membership_and_errors(project,tmp_path):
    (project/'opensfm/features/sub__a.jpg.features.npz').unlink()
    (project/'images/sub/a.jpg').unlink()
    client,out=open_client(project,tmp_path)
    assert client.get('/api/project').json['stats']['observations']==6
    assert client.get('/api/scene/0/point/0').json['stats']['count']==3
    data=client.get('/api/images/0/features').json
    assert data['points']==[] and data['resource_error']
    assert len(data['observations'])==2 and any(o['retained'] for o in data['observations'])


def test_ambiguous_components_do_not_duplicate_final_observations(project,tmp_path):
    recfile=project/'opensfm/reconstruction.topocentric.json'
    rec=json.loads(recfile.read_text());rec.append(rec[0])
    recfile.write_text(json.dumps(rec))
    client,out=open_client(project,tmp_path)
    for c in [0,1]:
        point=client.get(f'/api/scene/{c}/point/0').json
        assert point['ambiguous'] and point['stats']['mean'] is None
        assert point['observations']==[]
        comparison=client.get(f'/api/comparison/summary?component={c}').json
        assert not comparison['available'] and comparison['states']=={'unknown':2}
    assert client.get('/api/comparison/summary?scope=project').json['retained_observations']==3


def test_final_new_track_and_unregistered_scope(project,tmp_path):
    sfm=project/'opensfm';recfile=sfm/'reconstruction.topocentric.json'
    rec=json.loads(recfile.read_text());rec[0]['points']['new']={'coordinates':[0,0,5],'color':[1,2,3]}
    del rec[0]['shots']['sub/c.jpg'];recfile.write_text(json.dumps(rec))
    raw,final=final_subset(project)
    for name in ['sub/a.jpg','sub/b.jpg']:
        final.add_observation(name,'new',raw.get_shot_observations(name)['1'])
    final.write_to_file(str(sfm/'tracks_reconstructed.csv'))
    client,out=open_client(project,tmp_path)
    assert client.get('/api/tracks/new').json['final_length']==2
    assert client.get('/api/comparison/tracks?state=added').json['items'][0]['track']=='new'
    pair=client.get('/api/comparison/pairs/0/1?state=final').json
    assert set(pair['tracks'])=={'0','new'}
    assert client.get('/api/comparison/summary').json['raw_observations']==4
    assert client.get('/api/comparison/summary?scope=project').json['raw_observations']==6
    assert client.get('/api/comparison/images?scope=project&unregistered=1').json['total']==1
    assert client.get('/api/comparison/pairs/0/2').status_code==404
    assert client.get('/api/comparison/pairs/0/2?scope=project').status_code==200


def test_capture_order_and_final_content_fingerprint(project,tmp_path):
    sfm=project/'opensfm';(sfm/'exif/sub').mkdir(parents=True)
    for name,stamp in [('a',300),('b',100),('c',200)]:
        (sfm/f'exif/sub__{name}.jpg.exif').write_text(json.dumps({'capture_time':stamp}))
    client,out=open_client(project,tmp_path)
    info=client.get('/api/comparison/summary').json
    assert info['order_label']=='拍摄时间'
    assert [i['image'] for i in info['timeline']]==[1,2,0]
    file=sfm/'tracks_reconstructed.csv';stat=file.stat();before=fingerprint(project)
    blob=file.read_bytes();file.write_bytes(blob[:-1]+bytes([blob[-1]^1]))
    os.utime(file,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert fingerprint(project)!=before
    file.write_bytes(blob)
    assert client.get('/api/comparison/summary?edge_threshold=0').status_code==400
    assert client.get('/api/comparison/summary?scope=invalid').status_code==400


def test_camera_relations_distinguish_direct_final_and_outside(project,tmp_path):
    client,out=open_client(project,tmp_path)
    data=client.get('/api/images/0/relations?component=0').json
    assert data['final_available'] and data['selected_registered']
    items={r['id']:r for r in data['items']}
    assert items[1]['kind']=='both' and items[1]['direct_count']==2
    assert items[2]['kind']=='final' and items[2]['final_count']==1
    # A final common Track does not require a saved direct matching pair.
    assert items[2]['direct_count']==0
    with sqlite3.connect(out/'index.sqlite') as db:
        db.execute('UPDATE pair_changes SET final_count=0 WHERE scope=0 AND a=0 AND b=1')
    items={r['id']:r for r in client.get('/api/images/0/relations').json['items']}
    assert items[1]['kind']=='direct'
    with sqlite3.connect(out/'index.sqlite') as db:
        db.execute('DELETE FROM shots WHERE component=0 AND image=1')
    items={r['id']:r for r in client.get('/api/images/0/relations').json['items']}
    assert not items[1]['in_component'] and items[1]['final_count'] is None
    assert items[1]['kind']=='unknown'


def test_camera_relations_missing_final_does_not_claim_direct_only(project,tmp_path):
    (project/'opensfm/tracks_reconstructed.csv').unlink()
    client,out=open_client(project,tmp_path)
    data=client.get('/api/images/0/relations').json
    assert not data['final_available']
    assert data['items'][0]['kind']=='unknown'
    assert data['items'][0]['final_count'] is None


def test_matching_relations_ignore_final_tracks_and_include_outside(project,tmp_path):
    (project/'opensfm/tracks_reconstructed.csv').unlink()
    client,out=open_client(project,tmp_path)
    data=client.get('/api/images/0/matching-relations?component=0').json
    assert data['selected_registered']
    assert data['items']==[dict(id=1,name='sub/b.jpg',direct_count=2,in_component=True)]
    with sqlite3.connect(out/'index.sqlite') as db:
        db.execute('DELETE FROM shots WHERE component=0 AND image=1')
    data=client.get('/api/images/0/matching-relations?component=0').json
    assert data['items'][0]['in_component'] is False
    assert data['items'][0]['direct_count']==2
    assert client.get('/api/images/2/matching-relations').json['items']==[]


def test_retention_sort_places_unavailable_last(project, tmp_path):
    final_subset(project)
    client, out = open_client(project, tmp_path)
    with sqlite3.connect(out / 'index.sqlite') as connection:
        connection.execute('UPDATE image_changes SET raw_count=0, retained=0 WHERE image=0')
    response = client.get('/api/comparison/images?component=0&sort=retention')
    assert response.status_code == 200
    items = response.json['items']
    assert items[-1]['image'] == 0
    rates = [item['retained'] / item['raw_count'] for item in items[:-1]]
    assert rates == sorted(rates)
