"""Build disposable viewer indexes in a separate process."""
import argparse
import json
import sqlite3
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from overlap import component_overlap
from track_comparison import read_sources, build_final, build_comparison
from core import (atomic_json, project_paths, image_paths, feature_path, load_points,
                  read_matches, canonical_pair, fingerprint, summary, reprojections)

DDL = '''
CREATE TABLE images(id INTEGER PRIMARY KEY,name TEXT UNIQUE,path TEXT,width INT,height INT,features INT,feature_path TEXT,problem TEXT,neighbors INT DEFAULT 0,matches INT DEFAULT 0,capture_time REAL,input_order INT);
CREATE TABLE pairs(a INT,b INT,count INT,invalid INT,data BLOB,PRIMARY KEY(a,b));
CREATE INDEX pair_reverse ON pairs(b);
CREATE TABLE tracks(id TEXT PRIMARY KEY,length INT);
CREATE INDEX track_length ON tracks(length);
CREATE TABLE observations(track TEXT,image INT,feature INT,x REAL,y REAL,PRIMARY KEY(track,image));
CREATE INDEX obs_feature ON observations(image,feature);
CREATE TABLE points(component INT,idx INT,track TEXT,x REAL,y REAL,z REAL,PRIMARY KEY(component,idx),UNIQUE(component,track));
CREATE INDEX point_track ON points(track);
CREATE TABLE shots(component INT,image INT,PRIMARY KEY(component,image));
CREATE TABLE errors(component INT,track TEXT,image INT,error REAL,px REAL,py REAL,state TEXT,PRIMARY KEY(component,track,image));
CREATE INDEX error_image ON errors(component,image,error);
CREATE INDEX error_value ON errors(component,error);
CREATE TABLE image_stats(component INT,image INT,mean REAL,median REAL,p95 REAL,max REAL,count INT,PRIMARY KEY(component,image));
CREATE TABLE track_stats(component INT,track TEXT,mean REAL,median REAL,p95 REAL,max REAL,count INT,PRIMARY KEY(component,track));
CREATE INDEX track_error ON track_stats(component,max);
CREATE TABLE point_stats(component INT,idx INT,mean REAL,median REAL,p95 REAL,max REAL,rmse REAL,count INT,PRIMARY KEY(component,idx));
CREATE INDEX point_error ON point_stats(component,mean);
'''


def camera_from_json(value):
    from opensfm import pygeometry
    kind = value.get('projection_type', 'perspective')
    if kind == 'brown':
        cam = pygeometry.Camera.create_brown(value['focal_x'], value['focal_y'] / value['focal_x'],
            np.array([value.get('c_x', 0), value.get('c_y', 0)], dtype=float),
            np.array([value.get(k, 0) for k in ['k1', 'k2', 'k3', 'p1', 'p2']], dtype=float))
    elif kind == 'perspective':
        cam = pygeometry.Camera.create_perspective(value['focal'], value.get('k1', 0), value.get('k2', 0))
    else:
        raise ValueError('尚未验证的相机模型: ' + kind)
    cam.width, cam.height = int(value['width']), int(value['height'])
    return cam


def build(project, out, reconstruction=None):
    from opensfm import pymap, pygeometry
    project, sfm, recfile = project_paths(project, reconstruction)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    before = fingerprint(project, reconstruction)
    warnings = []
    def progress(stage, percent):
        atomic_json(out / 'status.json', dict(stage=stage, percent=percent, state='building'))
    progress('读取项目与图像', 2)
    recs = json.loads(recfile.read_text()) if recfile.exists() else []
    if not isinstance(recs, list):
        raise ValueError('重建文件必须是分量数组')
    managers, sources = read_sources(sfm)
    names = image_paths(project, sfm)
    input_order = {name:i for i,name in enumerate(names)}
    for manager in managers.values():
        if manager is not None:
            for name in manager.get_shot_ids():
                names.setdefault(name, str(project / 'images' / name))
    for rec in recs:
        for name in rec.get('shots', {}):
            names.setdefault(name, str(project / 'images' / name))
    db = sqlite3.connect(str(out / 'index.sqlite'))
    db.executescript(DDL)
    ids = {}; counts = {}; dimensions = {}
    for i, (name, path) in enumerate(sorted(names.items())):
        ids[name] = i
        problems = []
        w = h = 0
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception as e:
            problems.append('图像不可读: ' + str(e))
            exif = sfm / 'exif' / (name.replace('/', '__') + '.exif')
            if exif.exists():
                meta = json.loads(exif.read_text()); w = meta.get('width', 0); h = meta.get('height', 0)
        fp = feature_path(sfm, name)
        try:
            n = len(load_points(fp))
        except Exception as e:
            n = 0; problems.append('特征不可读: ' + str(e))
        capture_time = None
        exif = sfm / 'exif' / (name.replace('/', '__') + '.exif')
        if exif.exists():
            try:
                stamp = json.loads(exif.read_text()).get('capture_time')
                if isinstance(stamp, (float,int)) and np.isfinite(stamp) and stamp > 0: capture_time=float(stamp)
            except (ValueError, OSError): pass
        counts[i] = n; dimensions[i] = (w, h)
        db.execute('INSERT INTO images(id,name,path,width,height,features,feature_path,problem,capture_time,input_order) VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (i,name,path,w,h,n,str(fp),'; '.join(problems),capture_time,input_order.get(name)))
        if i % 100 == 0:
            progress('读取二维特征 %d / %d' % (i, len(names)), 3 + 17*i/max(len(names),1))
    db.commit()
    progress('汇总双向匹配', 21)
    matchfiles = list((sfm / 'matches').glob('*_matches.pkl.gz'))
    owners = {name.replace('/', '__') + '_matches.pkl.gz': i for name,i in ids.items()}
    for k, file in enumerate(matchfiles):
        a = owners.get(file.name)
        if a is None:
            warnings.append('匹配文件没有对应图像: ' + file.name); continue
        try:
            matches = read_matches(file)
            for name, raw in matches.items():
                b = ids.get(name)
                if b is None or b == a:
                    continue
                aa, bb, pairs = canonical_pair(a,b,raw)
                ok = ((pairs[:,0]>=0)&(pairs[:,0]<counts[aa])&(pairs[:,1]>=0)&(pairs[:,1]<counts[bb]))
                invalid = int((~ok).sum()); pairs = pairs[ok]
                old = db.execute('SELECT data,invalid FROM pairs WHERE a=? AND b=?',(aa,bb)).fetchone()
                if old:
                    pairs = np.concatenate([np.frombuffer(old[0], dtype='<i4').reshape(-1,2),pairs]); invalid += old[1]
                pairs = np.unique(pairs, axis=0).astype('<i4')
                db.execute('INSERT OR REPLACE INTO pairs VALUES(?,?,?,?,?)',(aa,bb,len(pairs),invalid,pairs.tobytes()))
        except Exception as e:
            warnings.append(file.name + ': ' + str(e))
        if k % 100 == 0:
            progress('汇总匹配 %d / %d' % (k,len(matchfiles)),21 + 9*k/max(len(matchfiles),1))
    db.execute('UPDATE images SET neighbors=(SELECT count(*) FROM pairs WHERE (a=images.id OR b=images.id) AND count>0),matches=COALESCE((SELECT sum(count) FROM pairs WHERE a=images.id OR b=images.id),0)')
    db.commit()
    progress('解析二进制 Tracks', 32)
    manager = managers['matched']
    for source,meta in sources.items():
        if meta['status']!='ready': warnings.append(source+' Tracks: '+meta['status'])
    track_counts = Counter(); batch = []
    if manager:
        for k, name in enumerate(manager.get_shot_ids()):
            if name not in ids:
                warnings.append('track 引用未知图像: '+name); continue
            image = ids[name]
            for track, obs in manager.get_shot_observations(name).items():
                x,y = [float(v) if np.isfinite(v) else None for v in obs.point]
                batch.append((track,image,obs.id,x,y))
                track_counts[track] += 1
            if len(batch) > 20000:
                db.executemany('INSERT INTO observations VALUES(?,?,?,?,?)',batch); batch=[]
            if k % 100 == 0:
                progress('建立观测索引 %d / %d' % (k,len(names)),35+20*k/max(len(names),1))
        db.executemany('INSERT INTO observations VALUES(?,?,?,?,?)',batch)
        db.executemany('INSERT INTO tracks VALUES(?,?)',track_counts.items())
        db.commit()
    scenes=[]; error_summaries=[]
    for component,rec in enumerate(recs):
        progress('准备三维分量 %d' % component,56)
        pts=[]; point_rows=[]
        for track,point in sorted(rec.get('points',{}).items()):
            xyz = np.asarray(point['coordinates'], dtype=float)
            if not np.isfinite(xyz).all(): continue
            color = np.clip(point.get('color',[180,180,180]),0,255)
            idx=len(pts); pts.append([*xyz,*color]); point_rows.append((component,idx,track,*xyz.tolist()))
        db.executemany('INSERT INTO points VALUES(?,?,?,?,?,?)',point_rows)
        pts=np.asarray(pts,dtype=float).reshape(-1,6)
        origin=np.median(pts[:,:3],axis=0) if len(pts) else np.zeros(3)
        bounds=np.percentile(pts[:,:3],[1,99],axis=0) if len(pts) else np.array([[-1]*3,[1]*3])
        scale=max(float(np.linalg.norm(bounds[1]-bounds[0])),10)
        packed=pts.copy(); packed[:,:3]-=origin; packed[:,3:]/=255
        packed.astype('<f4').tofile(out / ('points-%d.bin'%component))
        camera_models={}; unsupported={}
        for name,value in rec.get('cameras',{}).items():
            try: camera_models[name]=camera_from_json(value)
            except ValueError as e: unsupported[name]=str(e); warnings.append(str(e))
        cameras=[]
        for k,(name,shot) in enumerate(rec.get('shots',{}).items()):
            if name not in ids: continue
            image=ids[name]; pose=pygeometry.Pose(shot['rotation'],shot['translation'])
            center=pose.get_origin(); rotation=pose.get_rotation_matrix().T
            camera=camera_models.get(shot['camera']); rays=[]
            if camera:
                w,h=camera.width,camera.height; size=max(w,h)
                for u,v in [(0,0),(w-1,0),(w-1,h-1),(0,h-1)]:
                    bearing=camera.pixel_bearing(np.array([(u+.5-w/2)/size,(v+.5-h/2)/size]))
                    rays.append((rotation @ bearing).tolist())
            cameras.append(dict(id=image,name=name,center=(center-origin).tolist(),world_center=center.tolist(),
                                rotation=rotation.tolist(),rays=rays,model=shot['camera'],supported=camera is not None))
            db.execute('INSERT INTO shots VALUES(?,?)',(component,image))
            records=db.execute('SELECT o.track,o.x,o.y,p.x,p.y,p.z FROM observations o LEFT JOIN points p ON p.track=o.track AND p.component=? WHERE o.image=?',(component,image)).fetchall()
            valid=[r for r in records if r[3] is not None]
            err_rows=[(component,r[0],image,None,None,None,'no_point') for r in records if r[3] is None]
            if valid and camera:
                values=reprojections(camera,pose,np.array([r[3:] for r in valid]),np.array([r[1:3] for r in valid],dtype=float))
                err_rows += [(component,r[0],image,*value) for r,value in zip(valid,values)]
            elif valid:
                err_rows += [(component,r[0],image,None,None,None,'unsupported_camera') for r in valid]
            db.executemany('INSERT INTO errors VALUES(?,?,?,?,?,?,?)',err_rows)
            st=summary([r[3] for r in err_rows if r[3] is not None])
            db.execute('INSERT INTO image_stats VALUES(?,?,?,?,?,?,?)',(component,image,st['mean'],st['median'],st['p95'],st['max'],st['count']))
            if k % 50 == 0:
                progress('重投影计算 · 分量 %d · 图像 %d / %d'%(component,k,len(rec['shots'])),58+27*(component+k/max(len(rec['shots']),1))/max(len(recs),1))
        scenes.append(dict(id=component,points=len(pts),cameras=cameras,origin=origin.tolist(),extent=scale,
                           bounds=(bounds-origin).tolist(),reference=rec.get('reference_lla'),
                           coordinates='局部 ENU' if 'topocentric' in recfile.name else '重建文件坐标'))
        db.commit()
        progress('汇总重建点误差',87)
        batch=[]; point_summaries={}; current=None; values=[]
        def flush_track(t,vals):
            if t is None: return
            st=summary(vals)
            point_summaries[t]=st
            batch.append((component,t,st['mean'],st['median'],st['p95'],st['max'],st['count']))
        for track,error in db.execute('SELECT track,error FROM errors WHERE component=? AND state="valid" ORDER BY track',(component,)):
            if track!=current:
                flush_track(current,values); current=track; values=[]
            values.append(error)
        flush_track(current,values)
        db.executemany('INSERT INTO track_stats VALUES(?,?,?,?,?,?,?)',batch)
        point_batch=[]
        for idx,track in db.execute('SELECT idx,track FROM points WHERE component=? ORDER BY idx',(component,)):
            st=point_summaries.get(track) or summary([])
            point_batch.append((component,idx,st['mean'],st['median'],st['p95'],st['max'],st['rmse'],st['count']))
        db.executemany('INSERT INTO point_stats VALUES(?,?,?,?,?,?,?,?)',point_batch)
        # Float64 retains extreme finite residuals; NaN denotes unavailable, never zero.
        np.asarray([r[2] if r[2] is not None else np.nan for r in point_batch],dtype='<f8').tofile(out/('point-errors-%d.bin'%component))
        point_summary=summary([r[2] for r in point_batch if r[2] is not None])
        point_summary['points']=len(pts)
        point_summary['unavailable']=len(pts)-point_summary['count']
        point_summary['above_threshold']=sum(r[2] is not None and r[2]>=3 for r in point_batch)
        point_summary['metric']='mean_per_point_px'
        error_summaries.append(point_summary)
        db.commit()
    progress('计算最终保留观测误差',88)
    error_summaries = build_final(db,recs,ids,managers['final'],sources,out,camera_from_json,warnings)
    progress('比较 Track 与连接变化',89)
    build_comparison(db,recs,sources,out)
    managers.clear()
    overlap_summaries=[]
    for component in range(len(recs)):
        progress('计算共视重叠 · 分量 %d' % (component+1), 90+4*component/max(len(recs),1))
        overlap_summaries.append(component_overlap(db,component))
    progress('完成统计',95)
    lengths=list(track_counts.values()); hist=sorted(Counter(lengths).items())
    stats=dict(images=len(ids),features=sum(counts.values()),tracks=len(track_counts),observations=sum(lengths),
               track_images=db.execute('SELECT count(DISTINCT image) FROM observations').fetchone()[0],
               track_lengths=summary(lengths),histogram=hist,
               matched_pairs=db.execute('SELECT count(*) FROM pairs WHERE count>0').fetchone()[0],
               empty_pairs=db.execute('SELECT count(*) FROM pairs WHERE count=0').fetchone()[0],
               invalid_matches=db.execute('SELECT COALESCE(sum(invalid),0) FROM pairs').fetchone()[0],
               reconstructed_tracks=db.execute('SELECT count(*) FROM tracks WHERE EXISTS(SELECT 1 FROM points WHERE points.track=tracks.id)').fetchone()[0],
               errors=error_summaries,overlap=overlap_summaries,sources=sources)
    db.execute('ANALYZE'); db.commit(); db.close()
    if fingerprint(project,reconstruction)!=before:
        raise RuntimeError('索引期间源文件发生变化，请等待任务完成后重新索引')
    atomic_json(out/'scene.json',scenes)
    atomic_json(out/'summary.json',dict(project=str(project),name=project.name,reconstruction=str(recfile),
                                       fingerprint=before,stats=stats,warnings=warnings[:200],warning_count=len(warnings)))
    atomic_json(out/'status.json',dict(state='ready',stage='索引完成',percent=100))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',required=True); parser.add_argument('--out',required=True); parser.add_argument('--reconstruction')
    args=parser.parse_args()
    try: build(args.project,args.out,args.reconstruction)
    except Exception as e:
        traceback.print_exc()
        atomic_json(Path(args.out)/'status.json',dict(state='error',stage=str(e),percent=0))
        raise
