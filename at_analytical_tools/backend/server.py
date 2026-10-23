import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, g, jsonify, request, send_file, send_from_directory
from PIL import Image
from werkzeug.exceptions import HTTPException
from feature_stages import feature_stages
from replay import register_replay_api
from comparison_api import register_comparison_api, observation_details, source_info
from core import ROOT, atomic_json, fingerprint, load_points, project_paths, pixel_coordinates


class ProjectStore:
    def __init__(self, project, cache, reconstruction=None, rebuild=False):
        self.project, self.sfm, self.rec = project_paths(project, reconstruction)
        self.reconstruction = reconstruction
        key = hashlib.sha256((str(self.project)+'|'+str(self.rec)).encode()).hexdigest()[:20]
        self.root = Path(cache).resolve() / key
        self.root.mkdir(parents=True, exist_ok=True)
        self.active = None; self.pending = None; self.proc = None; self.log = None
        self.lock = threading.RLock()
        self.state = dict(state='building',stage='检查缓存',percent=0)
        self.initial_rebuild=rebuild
        threading.Thread(target=self.initialize,daemon=True).start()

    def initialize(self):
        try:
            current = self.root / 'current.json'
            if not self.initial_rebuild:
                candidates = [self.root / json.loads(current.read_text())['generation']] if current.exists() else sorted(self.root.glob('generation-*'),key=lambda p:p.stat().st_mtime,reverse=True)
                expected = fingerprint(self.project,self.reconstruction)
                for folder in candidates:
                    if not (folder/'summary.json').exists(): continue
                    meta = json.loads((folder/'summary.json').read_text())
                    if meta['fingerprint'] == expected:
                        self.active=folder; self.state=dict(state='ready',stage='缓存已就绪',percent=100)
                        atomic_json(current,dict(generation=folder.name))
                        return
            self.start()
        except Exception as e:
            self.state=dict(state='error',stage=str(e),percent=0)

    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None: return
            folder=self.root/('generation-'+uuid.uuid4().hex)
            folder.mkdir()
            self.pending=folder
            self.state=dict(state='building',stage='开始准备索引',percent=0)
            atomic_json(folder/'status.json',self.state)
            self.log=open(folder/'build.log','w')
            cmd=[sys.executable,'-B',str(Path(__file__).with_name('indexer.py')),'--project',str(self.project),'--out',str(folder)]
            if self.reconstruction: cmd+=['--reconstruction',self.reconstruction]
            self.proc=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,env=os.environ.copy())

    def status(self):
        with self.lock:
            if self.pending:
                file=self.pending/'status.json'
                if file.exists(): self.state=json.loads(file.read_text())
                code=self.proc.poll()
                if code is not None:
                    if self.state['state']=='ready' and code==0:
                        self.active=self.pending
                        atomic_json(self.root/'current.json',dict(generation=self.active.name))
                    elif self.state['state']!='error':
                        self.state=dict(state='error',stage='索引进程失败，请查看 '+str(self.pending/'build.log'),percent=0)
                    self.pending=None
                    if self.log: self.log.close(); self.log=None
                elif self.state['state']=='ready':
                    self.state=dict(state='building',stage='释放索引资源',percent=99)
            return {**self.state,'available':self.active is not None,'generation':self.active.name if self.active else None,
                    'project':str(self.project),'cache':str(self.root)}


@lru_cache(maxsize=24)
def cached_points(path, mtime):
    return load_points(path)


def create_app(store):
    app=Flask(__name__,static_folder=None)
    app.config['MAX_CONTENT_LENGTH']=1024
    register_comparison_api(app)
    register_replay_api(app)
    dist=Path(__file__).parent.parent/'frontend'/'dist'

    @app.errorhandler(Exception)
    def handle_error(error):
        code=error.code if isinstance(error,HTTPException) else (400 if isinstance(error,(ValueError,TypeError)) else 500)
        if code==500: app.logger.exception('request failed')
        return jsonify(error=str(error)),code

    @app.before_request
    def snapshot():
        if request.path.startswith('/api/') and request.path not in ('/api/status','/api/reindex'):
            store.status()
            if not store.active: abort(503,description='索引尚未就绪')
            g.folder=store.active
            expected=request.args.get('generation')
            if expected and expected!=g.folder.name: abort(409,description='数据版本已更新，请刷新页面')
            g.db=sqlite3.connect('file:'+str(g.folder/'index.sqlite')+'?mode=ro',uri=True)
            g.db.row_factory=sqlite3.Row

    @app.teardown_request
    def close_db(error):
        if hasattr(g,'db'): g.db.close()

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        if request.path.startswith('/api/') and response.mimetype=='application/json':
            response.headers['Cache-Control']='no-store'
        return response

    def one(sql,args=()):
        row=g.db.execute(sql,args).fetchone()
        if row is None: abort(404,description='记录不存在')
        return dict(row)

    def rows(sql,args=()): return [dict(r) for r in g.db.execute(sql,args)]
    def component(): return int(request.args.get('component',0))
    def pagination(): return min(max(int(request.args.get('limit',50)),1),200),max(int(request.args.get('offset',0)),0)
    def image(i): return one('SELECT * FROM images WHERE id=?',(i,))
    def points(im):
        path=Path(im['feature_path'])
        if not path.is_file(): abort(404,description='特征文件不存在')
        try: return cached_points(str(path),path.stat().st_mtime_ns)
        except Exception as e: abort(422,description='特征无法读取: '+str(e))
    def safe_list(array):
        return [[float(v) if np.isfinite(v) else None for v in row] for row in array]

    @app.get('/api/status')
    def status(): return jsonify(store.status())

    @app.post('/api/reindex')
    def reindex():
        # Reject cross-site forms; browsers must send a same-origin JSON request.
        if not request.is_json: abort(415)
        origin=request.headers.get('Origin')
        if origin and origin.rstrip('/')!=request.host_url.rstrip('/'): abort(403)
        store.start(); return jsonify(store.status()),202

    @app.get('/api/project')
    def project(): return send_file(g.folder/'summary.json')

    @app.get('/api/scenes')
    def scenes(): return send_file(g.folder/'scene.json')

    @app.get('/api/scene/<int:c>/points')
    def cloud(c):
        path=g.folder/('points-%d.bin'%c)
        if not path.exists(): abort(404)
        return send_file(path,mimetype='application/octet-stream',conditional=True)

    @app.get('/api/scene/<int:c>/point/<int:i>')
    def point(c,i):
        p=one('SELECT * FROM points WHERE component=? AND idx=?',(c,i))
        p['stats']=one('SELECT * FROM point_stats WHERE component=? AND idx=?',(c,i))
        observations=observation_details(c,track=p['track'])
        p['observations']=[o for o in observations if o['retained'] and o['fate']!='ambiguous']
        p['reference_observations']=[o for o in observations if not o['retained'] or o['fate']=='ambiguous']
        p['source']='final';p['source_status']=source_info()['final']['status']
        p['ambiguous']=bool(g.db.execute('SELECT 1 FROM ambiguous_points WHERE component=? AND track=?',(c,p['track'])).fetchone())
        return jsonify(p)

    @app.get('/api/scene/<int:c>/point-errors')
    def point_errors(c):
        path=g.folder/('point-errors-%d.bin'%c)
        if not path.exists(): abort(404)
        return send_file(path,mimetype='application/octet-stream',conditional=True)

    @app.get('/api/images')
    def images():
        limit,offset=pagination(); c=component()
        where=['1=1']; args=[]
        q=request.args.get('q','')
        if q: where.append('i.name LIKE ?');args.append('%'+q+'%')
        for param,column in [('min_features','i.features'),('min_neighbors','i.neighbors')]:
            if request.args.get(param): where.append(column+'>=?');args.append(int(request.args[param]))
        state=request.args.get('registered','all')
        if state!='all': where.append(('s.image IS NOT NULL' if state=='yes' else 's.image IS NULL'))
        if request.args.get('min_error'): where.append('e.max>=?');args.append(float(request.args['min_error']))
        source=' FROM images i LEFT JOIN image_stats e ON e.image=i.id AND e.component=? LEFT JOIN shots s ON s.image=i.id AND s.component=? WHERE '+' AND '.join(where)
        order={'name':'i.name','features':'i.features DESC','neighbors':'i.neighbors DESC','error':'e.max DESC'}.get(request.args.get('sort'),'i.name')
        total=g.db.execute('SELECT count(*)'+source,[c,c]+args).fetchone()[0]
        items=rows('SELECT i.id,i.name,i.width,i.height,i.features,i.problem,i.neighbors,i.matches,e.mean,e.max,e.p95,s.image IS NOT NULL AS registered'+source+' ORDER BY '+order+' LIMIT ? OFFSET ?',[c,c]+args+[limit,offset])
        return jsonify(items=items,total=total)

    @app.get('/api/images/<int:i>')
    def image_detail(i):
        im=image(i); im.pop('path');im.pop('feature_path')
        im['stats']=rows('SELECT * FROM image_stats WHERE image=? AND component=?',(i,component()))
        return jsonify(im)

    @app.get('/api/images/<int:i>/asset')
    def asset(i):
        im=image(i); path=Path(im['path'])
        if not path.is_file(): abort(404,description='图像文件不存在')
        size=request.args.get('size','preview')
        if size not in ('thumb','preview','original'): abort(400)
        cache=g.folder/'previews'; cache.mkdir(exist_ok=True)
        stamp=path.stat().st_mtime_ns
        dest=cache/('%d-%d-%s.jpg'%(i,stamp,size))
        if not dest.exists():
            try:
                with Image.open(path) as source:
                    source=source.convert('RGB') # Intentionally do not EXIF-transpose.
                    if size!='original': source.thumbnail((320,320) if size=='thumb' else (1600,1600))
                    tmp=cache/(dest.name+'.'+uuid.uuid4().hex+'.tmp')
                    source.save(tmp,format='JPEG',quality=90)
                    os.replace(str(tmp),str(dest))
            except Exception as e: abort(422,description='图像解码失败: '+str(e))
        return send_file(dest,mimetype='image/jpeg',conditional=True)

    @app.get('/api/images/<int:i>/features')
    def features(i):
        im=image(i)
        try: coords=points(im); resource_error=None
        except HTTPException as error: coords=np.empty((0,2));resource_error=error.description
        obs=observation_details(component(),image=i)
        sources=source_info()
        stages=feature_stages(obs,len(coords) if resource_error is None else None,sources['matched']['status'],sources['final']['status'])
        return jsonify(points=safe_list(coords),observations=obs,width=im['width'],height=im['height'],resource_error=resource_error,source_status=sources['final']['status'],input_status=sources['matched']['status'],stages=stages)

    @app.get('/api/images/<int:i>/matching-relations')
    def matching_relations(i):
        image(i)
        c=component()
        members={r[0] for r in g.db.execute('SELECT image FROM shots WHERE component=?',(c,))}
        items=rows('''SELECT im.id,im.name,p.count direct_count FROM pairs p
            JOIN images im ON im.id=CASE WHEN p.a=? THEN p.b ELSE p.a END
            WHERE (p.a=? OR p.b=?) AND p.count>0 ORDER BY p.count DESC,im.id''',(i,i,i))
        for item in items:
            item['in_component']=item['id'] in members
        return jsonify(items=items,selected_registered=i in members)

    @app.get('/api/images/<int:i>/relations')
    def camera_relations(i):
        image(i)
        c=component()
        members={r[0] for r in g.db.execute('SELECT image FROM shots WHERE component=?',(c,))}
        direct={r[0]:r[1] for r in g.db.execute('SELECT CASE WHEN a=? THEN b ELSE a END,count FROM pairs WHERE (a=? OR b=?) AND count>0',(i,i,i))}
        final={r[0]:r[1] for r in g.db.execute('SELECT CASE WHEN a=? THEN b ELSE a END,final_count FROM pair_changes WHERE scope=? AND (a=? OR b=?) AND final_count>0',(i,c,i,i))}
        info=json.loads((g.folder/('comparison-%d.json'%c)).read_text()) if (g.folder/('comparison-%d.json'%c)).exists() else {}
        ready=source_info()['final']['status']=='ready' and not info.get('ambiguity')
        items=[]
        for other in sorted(direct.keys() | final.keys()):
            name=g.db.execute('SELECT name FROM images WHERE id=?',(other,)).fetchone()[0]
            known=ready and i in members and other in members
            items.append(dict(id=other,name=name,direct_count=direct.get(other,0),final_count=final.get(other,0) if known else None,in_component=other in members,
                              kind=('both' if direct.get(other,0) and final.get(other,0) else 'direct' if direct.get(other,0) else 'final') if known else 'unknown'))
        return jsonify(items=items,final_available=ready,selected_registered=i in members)

    @app.get('/api/images/<int:i>/neighbor-ids')
    def neighbor_ids(i):
        image(i)
        if request.args.get('relation')=='final':
            return jsonify([r[0] for r in g.db.execute('SELECT CASE WHEN a=? THEN b ELSE a END FROM pair_changes WHERE scope=? AND (a=? OR b=?) AND final_count>0',(i,component(),i,i))])
        return jsonify([r[0] for r in g.db.execute('SELECT CASE WHEN a=? THEN b ELSE a END FROM pairs WHERE (a=? OR b=?) AND count>0',(i,i,i))])

    @app.get('/api/images/<int:i>/neighbors')
    def neighbors(i):
        image(i); limit,offset=pagination()
        where='(p.a=? OR p.b=?)'+('' if request.args.get('empty')=='1' else ' AND p.count>0')
        total=g.db.execute('SELECT count(*) FROM pairs p WHERE '+where,(i,i)).fetchone()[0]
        items=rows('SELECT i.id,i.name,p.count,p.invalid FROM pairs p JOIN images i ON i.id=CASE WHEN p.a=? THEN p.b ELSE p.a END WHERE '+where+' ORDER BY p.count DESC,i.name LIMIT ? OFFSET ?',(i,i,i,limit,offset))
        return jsonify(items=items,total=total)

    @app.get('/api/matches/<int:a>/<int:b>')
    def matches(a,b):
        ia,ib=image(a),image(b)
        row=one('SELECT * FROM pairs WHERE a=? AND b=?',(min(a,b),max(a,b)))
        pairs=np.frombuffer(row['data'],dtype='<i4').reshape(-1,2)
        if a>b:pairs=pairs[:,::-1]
        pa,pb=points(ia),points(ib)
        valid=(pairs[:,0]<len(pa))&(pairs[:,1]<len(pb))
        if not valid.all(): abort(409,description='特征与匹配不一致，请重新索引')
        return jsonify(a=a,b=b,count=len(pairs),invalid=row['invalid'],pairs=pairs.tolist(),a_points=safe_list(pa[pairs[:,0]]),b_points=safe_list(pb[pairs[:,1]]),
                       name_a=ia['name'],name_b=ib['name'],width_a=ia['width'],height_a=ia['height'],width_b=ib['width'],height_b=ib['height'])

    @app.get('/api/tracks')
    def tracks():
        c=component(); limit,offset=pagination(); where=['1=1'];args=[c,c]
        if request.args.get('q'):where.append('t.id LIKE ?');args.append('%'+request.args['q']+'%')
        for param,op in [('min_length','>='),('max_length','<=')]:
            if request.args.get(param):where.append('t.length'+op+'?');args.append(int(request.args[param]))
        state=request.args.get('reconstructed','all')
        if state!='all':where.append('p.idx IS '+('NOT NULL' if state=='yes' else 'NULL'))
        if request.args.get('min_error'):where.append('e.max>=?');args.append(float(request.args['min_error']))
        source=' FROM tracks t LEFT JOIN points p ON p.track=t.id AND p.component=? LEFT JOIN track_stats e ON e.track=t.id AND e.component=? WHERE '+' AND '.join(where)
        order={'length':'t.length DESC','error':'e.max DESC','short':'t.length ASC'}.get(request.args.get('sort'),'t.length DESC')
        total=g.db.execute('SELECT count(*)'+source,args).fetchone()[0]
        return jsonify(total=total,items=rows('SELECT t.*,p.idx,e.mean,e.median,e.p95,e.max'+source+' ORDER BY '+order+',t.id LIMIT ? OFFSET ?',args+[limit,offset]))

    @app.get('/api/tracks/<track>')
    def track_detail(track):
        c=component()
        change=one('SELECT * FROM track_changes WHERE scope=-1 AND track=?',(track,))
        t=dict(id=track,length=change['raw_length'],final_length=change['final_length'],change=change)
        t['component_change']=next(iter(rows('SELECT * FROM track_changes WHERE scope=? AND track=?',(c,track))),None)
        t['points']=rows('SELECT * FROM points WHERE track=?',(track,))
        t['stats']=rows('SELECT * FROM track_stats WHERE track=? AND component=?',(track,c))
        t['observations']=observation_details(c,track=track)
        t['source_status']=source_info()['final']['status']
        return jsonify(t)

    @app.get('/api/diagnostics')
    def diagnostics():
        c=component(); threshold=float(request.args.get('threshold',3));limit,offset=pagination()
        if not np.isfinite(threshold) or threshold<0: abort(400,description='误差阈值必须为非负有限数')
        where=['p.component=?'];args=[c]
        state=request.args.get('state','high')
        if state=='high': where.append('s.mean>=?');args.append(threshold)
        elif state=='unavailable': where.append('s.mean IS NULL')
        elif state!='all': abort(400,description='未知误差状态')
        q=request.args.get('q','')
        if q: where.append('p.track LIKE ?');args.append('%'+q+'%')
        order={'mean':'s.mean DESC','median':'s.median DESC','max':'s.max DESC','count':'s.count DESC'}.get(request.args.get('sort','mean'),'s.mean DESC')
        source=' FROM points p JOIN point_stats s ON s.component=p.component AND s.idx=p.idx WHERE '+' AND '.join(where)
        total=g.db.execute('SELECT count(*)'+source,args).fetchone()[0]
        records=rows('SELECT p.*,s.mean,s.median,s.p95,s.max,s.rmse,s.count'+source+' ORDER BY '+order+',p.idx LIMIT ? OFFSET ?',args+[limit,offset])
        above=g.db.execute('SELECT count(*) FROM point_stats WHERE component=? AND mean>=?',(c,threshold)).fetchone()[0]
        return jsonify(total=total,items=records,threshold=threshold,above_threshold=above,metric='mean_per_point_px',source='final',source_status=source_info()['final']['status'])

    @app.get('/')
    def index():
        if not (dist/'index.html').exists(): return '前端未构建，请执行 npm run build',503
        return send_file(dist/'index.html')

    @app.get('/assets/<path:name>')
    def static_assets(name): return send_from_directory(dist/'assets',name)
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser(description='只读空三结果查看器')
    p.add_argument('project');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8080)
    p.add_argument('--cache-dir',default=str(ROOT/'.cache'));p.add_argument('--reconstruction');p.add_argument('--reindex',action='store_true')
    args=p.parse_args()
    store=ProjectStore(args.project,args.cache_dir,args.reconstruction,args.reindex)
    app=create_app(store)
    print('AT Inspector: http://%s:%s'%(args.host,args.port),flush=True)
    from waitress import serve
    try:serve(app,host=args.host,port=args.port,threads=6)
    finally:
        if store.proc and store.proc.poll() is None: store.proc.terminate()
