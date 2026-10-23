"""Read-only, paginated comparison queries and shared observation provenance."""
import json
from collections import defaultdict
from flask import g, request, jsonify, abort


def rows(sql, args=()):
    return [dict(r) for r in g.db.execute(sql, args)]


def source_info():
    if not hasattr(g, 'track_sources'):
        g.track_sources = json.loads(
            (g.folder / 'summary.json').read_text())['stats']['sources']
    return g.track_sources


def comparison_scope():
    scope = request.args.get('scope', 'component')
    if scope not in ('component', 'project'): abort(400, description='未知比较范围')
    c = int(request.args.get('component', 0)) if scope == 'component' else -1
    path = g.folder / ('comparison-%d.json' % c)
    if not path.exists(): abort(404, description='分量不存在')
    return c, json.loads(path.read_text())


def pagination():
    return min(200, max(1, int(request.args.get('limit', 50)))), max(
        0, int(request.args.get('offset', 0)))


def observation_details(component, track=None, image=None):
    ready = source_info()['final']['status'] == 'ready'
    condition = 'o.track=?' if track is not None else 'o.image=?'
    value = track if track is not None else image
    query = '''SELECT o.track,o.image,o.feature,o.x,o.y,i.name,i.width,i.height,
        p.idx,e.error,e.px,e.py,e.state,r.error reference_error,r.px reference_px,r.py reference_py,r.state reference_state,
        f.feature final_feature,f.x final_x,f.y final_y,s.image registered,
        EXISTS(SELECT 1 FROM ambiguous_points ap WHERE ap.component=? AND ap.track=o.track) ambiguous
        FROM observations o JOIN images i ON i.id=o.image
        LEFT JOIN points p ON p.component=? AND p.track=o.track
        LEFT JOIN final_observations f ON f.component=? AND f.track=o.track AND f.image=o.image AND f.feature=o.feature
        LEFT JOIN errors e ON e.component=? AND e.track=f.track AND e.image=f.image
        LEFT JOIN reference_errors r ON r.component=? AND r.track=o.track AND r.image=o.image
        LEFT JOIN shots s ON s.component=? AND s.image=o.image WHERE ''' + condition
    result = rows(query, [component] * 6 + [value])
    for o in result:
        o['input_track'] = True
        o['input_x'], o['input_y'] = o['x'], o['y']
        o['retained'] = o.pop('final_feature') is not None
        o['coordinate_changed'] = o['retained'] and any(
            a is None or b is None or abs(a - b) > 1e-7
            for a, b in [(o['x'], o['final_x']), (o['y'], o['final_y'])])
        o['fate'] = ('unknown' if not ready else 'ambiguous' if o['ambiguous']
                     else 'retained' if o['retained'] else 'outside_component'
                     if o['registered'] is None else 'removed_observation'
                     if o['idx'] is not None else 'no_final_point')
        if o['retained']:
            o['x'], o['y'] = o['final_x'], o['final_y']
        o['state'] = o['state'] or o['fate']
    query = '''SELECT o.track,o.image,o.feature,o.x,o.y,i.name,i.width,i.height,p.idx,e.error,e.px,e.py,e.state,
        EXISTS(SELECT 1 FROM ambiguous_points ap WHERE ap.component=o.component AND ap.track=o.track) ambiguous
        FROM final_observations o JOIN images i ON i.id=o.image
        LEFT JOIN points p ON p.component=o.component AND p.track=o.track
        LEFT JOIN errors e ON e.component=o.component AND e.track=o.track AND e.image=o.image
        WHERE o.component=? AND ''' + condition + ''' AND NOT EXISTS(
            SELECT 1 FROM observations r WHERE r.track=o.track AND r.image=o.image AND r.feature=o.feature)'''
    for o in rows(query, (component, value)):
        o.update(input_track=False,
                 retained=True,
                 fate='ambiguous' if o['ambiguous'] else 'added',
                 coordinate_changed=False,
                 reference_error=None)
        result.append(o)
    return result


def register_comparison_api(app):

    @app.get('/api/comparison/summary')
    def overview():
        c, info = comparison_scope()
        threshold = int(request.args.get('edge_threshold', 20))
        if threshold < 1: abort(400, description='连接阈值必须大于零')
        images = rows(
            '''SELECT s.*,i.name,i.capture_time,i.input_order,e.mean,e.p95,e.max
            FROM image_changes s JOIN images i ON i.id=s.image
            LEFT JOIN image_stats e ON e.image=s.image AND e.component=s.scope WHERE s.scope=?''',
            (c, ))
        timed = bool(images) and all(i['capture_time'] is not None
                                     for i in images)
        ordered = all(i['input_order'] is not None for i in images)
        images.sort(key=lambda i: (i['capture_time'] if timed else i[
            'input_order'] if ordered else i['name'], i['image']))
        for n, i in enumerate(images):
            i['order'] = n + 1
            i['retention'] = i['retained'] / i['raw_count'] if info[
                'available'] and i['raw_count'] else None
            window = images[max(0, n - 2):min(len(images), n + 3)]
            count = sum(x['raw_count'] for x in window)
            i['smoothed'] = sum(
                x['retained'] or 0 for x in
                window) / count if info['available'] and count else None
        edges = rows('SELECT * FROM pair_changes WHERE scope=?', (c, ))

        def topology(column):
            graph = {i['image']: set() for i in images}
            count = 0
            for edge in edges:
                if (edge[column] or 0) >= threshold:
                    graph[edge['a']].add(edge['b'])
                    graph[edge['b']].add(edge['a'])
                    count += 1
            seen, groups = set(), 0
            for i in graph:
                if i in seen: continue
                groups += 1
                todo = [i]
                seen.add(i)
                while todo:
                    for neighbor in graph[todo.pop()]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            todo.append(neighbor)
            return dict(edges=count,
                        components=groups,
                        isolated=sum(not v for v in graph.values()),
                        degree={
                            k: len(v)
                            for k, v in graph.items()
                        })

        raw, final = topology('raw_count'), topology('final_count')
        for i in images:
            i['raw_neighbors'] = raw['degree'][i['image']]
            i['final_neighbors'] = final['degree'][
                i['image']] if info['available'] else None
        raw.pop('degree')
        final.pop('degree')
        info.update(
            timeline=images,
            order_label='拍摄时间' if timed else '输入顺序' if ordered else '名称顺序',
            topology=dict(
                threshold=threshold,
                matched=raw,
                final=final if info['available'] else None,
                weakened=sum(e['raw_count'] >= threshold and 0 <
                             (e['final_count'] or 0) < threshold
                             for e in edges) if info['available'] else None,
                vanished=sum(
                    e['raw_count'] >= threshold and e['final_count'] == 0
                    for e in edges) if info['available'] else None))
        info['clues'] = sorted(
            images, key=lambda i:
            (-(i['loss'] or 0), i['image']))[:5] if info['available'] else []
        return jsonify(info)

    @app.get('/api/comparison/images')
    def comparison_images():
        c, info = comparison_scope()
        limit, offset = pagination()
        q = request.args.get('q', '')
        where = 's.scope=? AND i.name LIKE ?'
        args = [c, '%' + q + '%']
        if request.args.get('unregistered') == '1':
            where += ' AND NOT EXISTS(SELECT 1 FROM shots sh WHERE sh.image=i.id)'
        order = {
            'loss': 's.loss DESC',
            'retention':
            '(s.raw_count=0 OR s.retained IS NULL),CAST(s.retained AS REAL)/NULLIF(s.raw_count,0)',
            'error': 'e.p95 DESC',
            'name': 'i.name'
        }.get(request.args.get('sort'), 's.loss DESC')
        source = ' FROM image_changes s JOIN images i ON i.id=s.image LEFT JOIN image_stats e ON e.image=s.image AND e.component=s.scope WHERE ' + where
        total = g.db.execute('SELECT count(*)' + source, args).fetchone()[0]
        items = rows(
            'SELECT s.*,i.name,i.width,i.height,e.mean,e.p95' + source +
            ' ORDER BY ' + order + ',i.id LIMIT ? OFFSET ?',
            args + [limit, offset])
        return jsonify(items=items, total=total, available=info['available'])

    @app.get('/api/comparison/tracks')
    def comparison_tracks():
        c, info = comparison_scope()
        limit, offset = pagination()
        where = ['scope=?', 'track LIKE ?']
        args = [c, '%' + request.args.get('q', '') + '%']
        state = request.args.get('state', 'all')
        if state != 'all':
            where.append('status=?')
            args.append(state)
        for param, column in [('raw_length', 'raw_length'),
                              ('final_length', 'final_length')]:
            if request.args.get(param):
                where.append(column + '=?')
                args.append(int(request.args[param]))
        if request.args.get('image'):
            where.append(
                '(EXISTS(SELECT 1 FROM observations o WHERE o.track=track_changes.track AND o.image=?) OR EXISTS(SELECT 1 FROM final_input f WHERE f.track=track_changes.track AND f.image=?))'
            )
            args.extend([int(request.args['image'])] * 2)
        source = ' FROM track_changes WHERE ' + ' AND '.join(where)
        order = {
            'loss': 'loss DESC',
            'length': 'raw_length DESC',
            'final': 'final_length DESC'
        }.get(request.args.get('sort'), 'loss DESC')
        total = g.db.execute('SELECT count(*)' + source, args).fetchone()[0]
        return jsonify(items=rows(
            'SELECT *' + source + ' ORDER BY ' + order +
            ',track LIMIT ? OFFSET ?', args + [limit, offset]),
                       total=total,
                       available=info['available'])

    @app.get('/api/comparison/pairs')
    def pairs():
        c, info = comparison_scope()
        limit, offset = pagination()
        threshold = int(request.args.get('edge_threshold', 20))
        if threshold < 1: abort(400)
        where = ['p.scope=?']
        args = [c]
        state = request.args.get('state', 'all')
        conditions = {
            'stable': 'p.final_count>=?',
            'weakened':
            'p.raw_count>=? AND p.final_count>0 AND p.final_count<?',
            'vanished': 'p.raw_count>=? AND p.final_count=0'
        }
        if state in conditions:
            where.append(conditions[state])
            args += [threshold] * conditions[state].count('?')
        if request.args.get('image'):
            where.append('(p.a=? OR p.b=?)')
            args += [int(request.args['image'])] * 2
        source = ' FROM pair_changes p JOIN images a ON a.id=p.a JOIN images b ON b.id=p.b WHERE ' + ' AND '.join(
            where)
        total = g.db.execute('SELECT count(*)' + source, args).fetchone()[0]
        return jsonify(items=rows(
            'SELECT p.*,a.name name_a,b.name name_b' + source +
            ' ORDER BY (p.raw_count-p.final_count) DESC,p.a,p.b LIMIT ? OFFSET ?',
            args + [limit, offset]),
                       total=total,
                       available=info['available'])

    @app.get('/api/comparison/pairs/<int:a>/<int:b>')
    def pair_detail(a, b):
        c, info = comparison_scope()
        limit, offset = pagination()
        table = 'final_input' if c < 0 else 'final_observations'
        cond = '' if c < 0 else ' AND fa.component=? AND fb.component=?'
        extra = [] if c < 0 else [c, c]
        # Two source branches also preserve changed feature associations.
        raw = '''SELECT ra.track,ra.feature af,rb.feature bf,CASE WHEN fa.track IS NOT NULL AND fb.track IS NOT NULL THEN fa.x ELSE ra.x END ax,
            CASE WHEN fa.track IS NOT NULL AND fb.track IS NOT NULL THEN fa.y ELSE ra.y END ay,
            CASE WHEN fa.track IS NOT NULL AND fb.track IS NOT NULL THEN fb.x ELSE rb.x END bx,
            CASE WHEN fa.track IS NOT NULL AND fb.track IS NOT NULL THEN fb.y ELSE rb.y END by,
            (fa.track IS NOT NULL AND fb.track IS NOT NULL) retained
            FROM observations ra JOIN observations rb ON rb.track=ra.track
            LEFT JOIN ''' + table + ''' fa ON fa.track=ra.track AND fa.image=ra.image AND fa.feature=ra.feature''' + (
            ' AND fa.component=?' if c >= 0 else '') + '''
            LEFT JOIN ''' + table + ''' fb ON fb.track=rb.track AND fb.image=rb.image AND fb.feature=rb.feature''' + (
                ' AND fb.component=?' if c >= 0 else '') + '''
            WHERE ra.image=? AND rb.image=?'''
        final = '''SELECT fa.track,fa.feature af,fb.feature bf,fa.x ax,fa.y ay,fb.x bx,fb.y by,1 retained
            FROM ''' + table + ''' fa JOIN ''' + table + ''' fb ON fb.track=fa.track
            WHERE fa.image=? AND fb.image=?''' + cond + ''' AND NOT EXISTS(SELECT 1 FROM observations ra JOIN observations rb ON ra.track=rb.track
                WHERE ra.track=fa.track AND ra.image=fa.image AND rb.image=fb.image AND ra.feature=fa.feature AND rb.feature=fb.feature)'''
        sql = '(' + raw + ' UNION ALL ' + final + ')'
        args = extra + [a, b, a, b] + extra
        state = request.args.get('state', 'all')
        where = ' WHERE retained=1' if state == 'final' else ' WHERE retained=0' if state == 'lost' and info[
            'available'] else ' WHERE 0' if state == 'lost' else ''
        if c >= 0:
            for image in [a, b]:
                if not g.db.execute(
                        'SELECT 1 FROM shots WHERE component=? AND image=?',
                    (c, image)).fetchone():
                    abort(404)
        count = g.db.execute('SELECT count(*) FROM ' + sql + where,
                             args).fetchone()[0]
        data = rows(
            'SELECT * FROM ' + sql + where +
            ' ORDER BY track LIMIT ? OFFSET ?', args + [limit, offset])
        images = rows('SELECT * FROM images WHERE id IN (?,?)', (a, b))
        byid = {
            i['id']: i
            for i in images
        }
        if a not in byid or b not in byid: abort(404)
        ia, ib = byid[a], byid[b]
        return jsonify(a=a,
                       b=b,
                       name_a=ia['name'],
                       name_b=ib['name'],
                       width_a=ia['width'],
                       height_a=ia['height'],
                       width_b=ib['width'],
                       height_b=ib['height'],
                       count=len(data),
                       total=count,
                       offset=offset,
                       invalid=0,
                       tracks=[r['track'] for r in data],
                       retained=[bool(r['retained']) for r in data],
                       pairs=[[r['af'], r['bf']] for r in data],
                       a_points=[[r['ax'], r['ay']] for r in data],
                       b_points=[[r['bx'], r['by']] for r in data],
                       available=info['available'])
