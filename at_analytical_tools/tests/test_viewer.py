import gzip
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core import canonical_pair, pixel_coordinates, read_matches, fingerprint, reprojections
from indexer import build, camera_from_json
from server import create_app
from opensfm import pymap, pygeometry


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root=tmp_path_factory.mktemp('at-fixture'); project=root/'project';sfm=project/'opensfm'
    for d in ['features','matches']: (sfm/d).mkdir(parents=True)
    (project/'images'/'sub').mkdir(parents=True)
    names=['sub/a.jpg','sub/b.jpg','sub/c.jpg'];listing=[]
    coords=[[[0,0],[.1,.1],[.2,.2]],[[-.2,0],[-.2,0],[.1,.1]],[[-.4,0],[.1,.1],[-.4,0]]]
    for i,name in enumerate(names):
        path=project/'images'/name;Image.new('RGB',(100,80),(30,80,60)).save(path);listing.append(str(path))
        np.savez_compressed(sfm/'features'/(name.replace('/','__')+'.features.npz'),points=np.c_[coords[i],np.ones((3,2))])
    (sfm/'image_list.txt').write_text('\n'.join(listing))
    for owner,m in [(names[1],{names[0]:np.array([[1,0],[2,1]])}),(names[0],{names[1]:np.array([[0,1],[1,2],[-1,0],[99,0]]),names[2]:[]})]:
        with gzip.open(sfm/'matches'/(owner.replace('/','__')+'_matches.pkl.gz'),'wb') as f:pickle.dump(m,f)
    manager=pymap.TracksManager()
    for i,name in enumerate(names):
        feature=[0,1,2][i];x,y=coords[i][feature]
        manager.add_observation(name,'0',pymap.Observation(x,y,.01,30,80,60,feature))
        feature=[1,2,0][i];x,y=coords[i][feature]
        manager.add_observation(name,'1',pymap.Observation(x,y,.01,30,80,60,feature))
    manager.write_to_file(str(sfm/'tracks.csv'))
    final=manager.construct_sub_tracks_manager(['0'],names)
    final.write_to_file(str(sfm/'tracks_reconstructed.csv'))
    camera={'projection_type':'brown','width':100,'height':80,'focal_x':1.,'focal_y':1.,'c_x':0.,'c_y':0.}
    rec=[{'cameras':{'cam':camera},'shots':{name:{'camera':'cam','rotation':[0,0,0],'translation':[-i,0,0]} for i,name in enumerate(names)},'points':{'0':{'coordinates':[0,0,5],'color':[30,80,60]}}}]
    (sfm/'reconstruction.topocentric.json').write_text(json.dumps(rec));out=root/'cache'
    before=fingerprint(project);build(project,out);assert fingerprint(project)==before
    return project,out


@pytest.fixture()
def client(dataset):
    project,out=dataset
    store=SimpleNamespace(active=out,status=lambda:{'state':'ready','available':True,'generation':out.name})
    return create_app(store).test_client()


def test_statistics_and_missing_point(client):
    stats=client.get('/api/project').json['stats']
    assert stats['tracks']==2 and stats['observations']==6
    assert stats['track_lengths']['mean']==3
    assert stats['reconstructed_tracks']==1
    assert 'no_point' not in stats['errors'][0]['states']
    assert stats['errors'][0]['points']==1
    assert stats['errors'][0]['count']==1
    assert stats['errors'][0]['max']<1e-5
    overlap=stats['overlap'][0]
    assert overlap['component']==0 and overlap['registered_images']==3
    assert overlap['mean'] is None and overlap['evaluated_images']==0
    assert overlap['metric']=='spatially_balanced_covisibility_final_v2'


def test_reverse_matching_dedup_and_bounds(client):
    a=client.get('/api/matches/0/1').json;b=client.get('/api/matches/1/0').json
    assert a['pairs']==[[0,1],[1,2]]
    assert b['pairs']==[[1,0],[2,1]]
    assert a['invalid']==2
    assert client.get('/api/images/0/neighbor-ids').json==[1]
    assert client.get('/api/images/1/neighbor-ids').json==[0]
    assert client.get('/api/images/0/neighbors?empty=1').json['total']==2
    assert client.get('/api/matches/0/2').json['count']==0
    assert client.get('/api/matches/1/2').status_code==404


def test_track_feature_and_point_links(client):
    point=client.get('/api/scene/0/point/0').json
    assert point['track']=='0'
    track=client.get('/api/tracks/0').json
    assert {o['feature'] for o in track['observations']}=={0,1,2}
    features=client.get('/api/images/1/features').json
    assert len(features['points'])==3
    assert next(o for o in features['observations'] if o['track']=='0')['feature']==1
    assert client.get('/api/tracks?min_length=4').json['total']==0
    assert client.get('/api/tracks?reconstructed=no').json['total']==1


def test_unknown_resources_and_generation(client):
    assert client.get('/api/images/999/asset').status_code==404
    assert client.get('/api/project?generation=wrong').status_code==409
    assert client.post('/api/reindex',data='{}').status_code==415
    assert client.post('/api/reindex',json={},headers={'Origin':'http://untrusted.test'}).status_code==403
    assert client.get('/api/images?min_features=bad').status_code==400
    assert client.get('/api/images/0/asset?size=../../etc/passwd').status_code==400


def test_image_cache_preserves_dimensions(client,dataset):
    p,out=dataset;before=fingerprint(p)
    response=client.get('/api/images/0/asset?size=original')
    assert response.status_code==200 and response.mimetype=='image/jpeg'
    assert fingerprint(p)==before
    assert (out/'previews').is_dir()


def test_projection_pixel_units_and_behind_camera():
    camera=camera_from_json({'projection_type':'perspective','focal':1,'width':100,'height':80})
    pose=pygeometry.Pose([0,0,0],[0,0,0])
    values=reprojections(camera,pose,[[0,0,5],[0,0,-5]],np.array([[.03,.04],[0,0]]))
    assert values[0][0]==pytest.approx(5)
    assert values[1][0] is None and values[1][3]=='behind_camera'
    assert pixel_coordinates([[0,0]],100,80).tolist()==[[49.5,39.5]]


def test_unsupported_camera_is_explicit():
    with pytest.raises(ValueError,match='尚未验证'):camera_from_json({'projection_type':'spherical'})


def test_restricted_pickle(tmp_path):
    class Payload:
        def __reduce__(self):return (eval,('1+1',))
    file=tmp_path/'bad.gz'
    with gzip.open(file,'wb') as f:pickle.dump(Payload(),f)
    with pytest.raises(pickle.UnpicklingError):read_matches(file)


def test_missing_feature_is_reported(client,dataset):
    project,out=dataset
    fp=project/'opensfm'/'features'/'sub__a.jpg.features.npz'
    hidden=fp.with_suffix('.hold');fp.rename(hidden)
    try:assert client.get('/api/images/0/features').json['resource_error']=='特征文件不存在'
    finally:hidden.rename(fp)


def test_invalid_threshold_and_large_indices(client):
    assert client.get('/api/diagnostics?threshold=nan').status_code==400
    assert client.get('/api/diagnostics?threshold=-1').status_code==400
    with pytest.raises(ValueError):canonical_pair(0,1,np.array([[2**40,0]],dtype=np.int64))


def test_point_diagnostics_excludes_unreconstructed_tracks(client):
    response=client.get('/api/diagnostics?state=all').json
    assert response['metric']=='mean_per_point_px'
    assert response['total']==1
    p=response['items'][0]
    assert p['track']=='0' and p['count']==3
    detail=client.get('/api/scene/0/point/0').json
    assert len(detail['observations'])==3
    assert detail['stats']['mean']==pytest.approx(p['mean'])
    assert client.get('/api/diagnostics?threshold=3').json['total']==0
    assert client.get('/api/diagnostics?state=unavailable').json['total']==0
    assert client.get('/api/diagnostics?state=all&q=absent').json['total']==0
    assert client.get('/api/diagnostics?state=bad').status_code==400
    cloud=np.frombuffer(client.get('/api/scene/0/point-errors').data,dtype='<f8')
    assert len(cloud)==1 and cloud[0]==pytest.approx(p['mean'])


def test_point_aggregation_and_unavailable(tmp_path):
    from core import summary
    stats=summary([3,4,12])
    assert stats['mean']==pytest.approx(19/3)
    assert stats['median']==4
    assert stats['rmse']==pytest.approx(np.sqrt(169/3))
    assert summary([])['mean'] is None
    assert summary([1,1e18])['max']==1e18


def test_unavailable_points_are_not_zero(dataset,tmp_path):
    import shutil
    project=tmp_path/'project'
    shutil.copytree(dataset[0],project)
    sfm=project/'opensfm'
    (sfm/'image_list.txt').write_text('\n'.join(str(project/'images'/name) for name in ['sub/a.jpg','sub/b.jpg','sub/c.jpg']))
    rec=json.loads((sfm/'reconstruction.topocentric.json').read_text())
    rec[0]['points']['1']={'coordinates':[0,0,-5],'color':[0,0,0]}
    rec[0]['points']['orphan']={'coordinates':[0,0,5],'color':[0,0,0]}
    (sfm/'reconstruction.topocentric.json').write_text(json.dumps(rec))
    raw=pymap.TracksManager.instanciate_from_file(str(sfm/'tracks.csv'))
    raw.write_to_file(str(sfm/'tracks_reconstructed.csv'))
    out=tmp_path/'index';build(project,out)
    store=SimpleNamespace(active=out,status=lambda:{'available':True})
    client=create_app(store).test_client()
    stats=client.get('/api/project').json['stats']['errors'][0]
    assert stats['points']==3 and stats['count']==1 and stats['unavailable']==2
    assert stats['states']['behind_camera']==3
    records=client.get('/api/diagnostics?state=unavailable').json['items']
    assert {p['track'] for p in records}=={'1','orphan'}
    assert all(p['mean'] is None and p['count']==0 for p in records)
    values=np.frombuffer(client.get('/api/scene/0/point-errors').data,dtype='<f8')
    assert np.isnan(values).sum()==2
