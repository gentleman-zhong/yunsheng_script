"""Read-only verification of the supplied project against the viewer cache."""
import json,sqlite3,sys,time
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core import fingerprint,load_points
root=Path(__file__).resolve().parents[1]
folder=Path(sys.argv[1]).resolve()
meta=json.loads((folder/'summary.json').read_text());s=meta['stats']
assert fingerprint(meta['project'],meta['reconstruction'])==meta['fingerprint'],'Source changed since index build'
con=sqlite3.connect('file:'+str(folder/'index.sqlite')+'?mode=ro',uri=True)
assert not con.execute('SELECT 1 FROM errors e LEFT JOIN final_observations f ON f.component=e.component AND f.track=e.track AND f.image=e.image WHERE f.track IS NULL LIMIT 1').fetchone()
# Independent aggregation checks, including the largest outlier and aligned color data.
records=con.execute('SELECT idx,mean,median,max,rmse,count FROM point_stats WHERE component=0 ORDER BY idx').fetchall()
assert len(records)==con.execute('SELECT count(*) FROM points WHERE component=0').fetchone()[0]
binary=np.fromfile(folder/'point-errors-0.bin',dtype='<f8')
assert len(binary)==len(records)
for idx,mean,median,maximum,rmse,count in records[::997]:
    track=con.execute('SELECT track FROM points WHERE component=0 AND idx=?',(idx,)).fetchone()[0]
    errors=np.array([r[0] for r in con.execute('SELECT error FROM errors WHERE component=0 AND track=? AND state="valid"',(track,))])
    assert count==len(errors)
    if count:
        assert np.isclose(mean,errors.mean()) and np.isclose(median,np.median(errors))
        assert np.isclose(rmse,np.sqrt(np.mean(errors**2))) and maximum==errors.max()
        assert binary[idx]==mean
    else: assert np.isnan(binary[idx])
means=np.array([r[1] for r in records if r[1] is not None])
assert np.isclose(np.median(means),s['errors'][0]['median'])
assert np.isclose(np.percentile(means,95),s['errors'][0]['p95'])
assert (means>=3).sum()==s['errors'][0]['above_threshold']
assert 'no_point' not in s['errors'][0]['states']
# Validate source projections independently using explicit Brown distortion equations.
from opensfm import pygeometry
rec=json.loads(Path(meta['reconstruction']).read_text())[0]
sample=con.execute('SELECT e.track,i.name,o.x,o.y,e.error FROM errors e JOIN images i ON i.id=e.image JOIN final_observations o ON o.component=e.component AND o.track=e.track AND o.image=e.image WHERE e.component=0 AND e.state="valid" ORDER BY e.error DESC LIMIT 30').fetchall()
for track,name,ox,oy,error in sample:
    shot=rec['shots'][name];camera=rec['cameras'][shot['camera']]
    assert camera['projection_type']=='brown'
    xyz=pygeometry.Pose(shot['rotation'],shot['translation']).transform(rec['points'][track]['coordinates'])
    x,y=xyz[:2]/xyz[2];r2=x*x+y*y
    radial=1+camera['k1']*r2+camera['k2']*r2*r2+camera['k3']*r2*r2*r2
    px=camera['focal_x']*(x*radial+2*camera['p1']*x*y+camera['p2']*(r2+2*x*x))+camera['c_x']
    py=camera['focal_y']*(y*radial+2*camera['p2']*x*y+camera['p1']*(r2+2*y*y))+camera['c_y']
    expected=np.hypot(px-ox,py-oy)*max(camera['width'],camera['height'])
    assert np.isclose(expected,error,rtol=1e-7,atol=1e-6),(track,expected,error)
report={'source_unchanged':fingerprint(meta['project'],meta['reconstruction'])==meta['fingerprint'],'points':len(records),'sample_point_aggregations':len(records[::997]),'source_brown_projections':len(sample),'point_mean_median_px':float(np.median(means)),'point_mean_p95_px':float(np.percentile(means,95)),'points_above_3_px':int((means>=3).sum())}
from check_overlap_real import check
report['overlap']=check(folder)
report['sources']=s['sources']
report['comparison']=json.loads((folder/'comparison-0.json').read_text())
(folder.parent/'real-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('comparison','sources')},ensure_ascii=False,indent=2))
