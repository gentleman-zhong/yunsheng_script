"""Read-only reconstruction order replay. Never infer historical poses or point clouds."""
import hashlib
import json
import re
from collections import OrderedDict, defaultdict
from threading import Lock
from pathlib import Path

from flask import abort, g, jsonify, request

PARSER_VERSION = 1
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
TIME = re.compile(r'^(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[,.]\d+)?)')
FAILURES = (
    'Could not find initial motion',
    'Initial motion did not generate enough points',
    'Re-triangulation after initial motion did not generate enough points')


def normalized(name):
    return name.strip().replace('\\', '/').removeprefix('./')


def split_names(value, known):
    # A file name itself may contain " and ". Resolve against known names first.
    if normalized(value) in known:
        return [value]
    parts = value.split(' and ')
    solutions = []

    def visit(start, names):
        if len(solutions) > 1:
            return
        if start == len(parts):
            solutions.append(names)
            return
        for end in range(start + 1, len(parts) + 1):
            name = ' and '.join(parts[start:end])
            if normalized(name) in known:
                visit(end, names + [name])

    if len(parts) < 20:
        visit(0, [])
    return solutions[0] if len(solutions) == 1 else parts


def branch(run, index):
    return dict(run_id=str(run),
                branch_id=str(index),
                events=[],
                initial_confirmed=False,
                complete=False,
                failed=False,
                warnings=[],
                summary_points=None)


def event(kind, names, line=None, timestamp=None, excerpt=None):
    return dict(kind=kind,
                names=names,
                line=line,
                timestamp=timestamp,
                excerpt=excerpt)


def parse_log(text, known):
    branches, active = [], None
    run = 0
    algorithm = None
    summaries = []

    def new_branch():
        nonlocal active
        active = branch(
            run or 1,
            sum(b['run_id'] == str(run or 1) for b in branches) + 1)
        branches.append(active)
        return active

    for number, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub('', raw).strip()
        stamp = TIME.match(line)
        timestamp = stamp.group(1).replace(',', '.') if stamp else None
        # Match emitted runtime markers, not config values printed before execution.
        if 'Starting incremental reconstruction' in line:
            run += 1
            algorithm = 'incremental'
            active = None
            summaries = []
            continue
        if re.search(
                r'Starting (?:triangulation|planar|global) reconstruction',
                line):
            run += 1
            algorithm = 'non_incremental'
            active = None
            continue
        match = re.search(
            r'(?:INFO:\s*|^)(Starting reconstruction with (.+))$', line)
        if match and algorithm != 'non_incremental':
            b = new_branch()
            b['events'].append(
                event('initial', split_names(match.group(2), known), number,
                      timestamp, line))
            continue
        if active and any(failure in line for failure in FAILURES):
            active['failed'] = True
            active['warnings'].append('初始对尝试失败：' + line[:400])
            continue
        match = re.search(r'(?:INFO:\s*|^)Adding (.+) to the reconstruction$',
                          line)
        if match and algorithm != 'non_incremental':
            if active is None or active['failed']:
                new_branch()
                active['warnings'].append('缺少已确认的初始对，仅能回放记录片段。')
            names = split_names(match.group(1), known)
            seen = {n for e in active['events'] for n in e['names']}
            fresh = [n for n in names if n not in seen]
            if len(fresh) != len(names):
                active['warnings'].append('重复加入记录已去重（行 %d）。' % number)
            if fresh:
                active['events'].append(
                    event('add', fresh, number, timestamp, line))
                active['initial_confirmed'] = active['events'][0][
                    'kind'] == 'initial' and len(
                        active['events'][0]['names']) == 2
        match = re.search(r'Reconstruction \d+: (\d+) images, (\d+) points',
                          line)
        if match:
            summaries.append((int(match.group(1)), int(match.group(2))))
        if re.search(r'\d+ partial reconstructions in total\.', line):
            current = [
                b for b in branches
                if b['run_id'] == str(run or 1) and not b['failed']
            ]
            for b in current:
                b['complete'] = True
            if len(current) == len(summaries) == 1:
                current[0]['summary_points'] = summaries[0][1]
    status = 'ready' if branches else 'non_incremental' if algorithm == 'non_incremental' else 'unsupported'
    return dict(branches=branches, status=status)


def parse_report(text):
    value = json.loads(text)
    records = value.get('reconstructions') if isinstance(value, dict) else None
    if not isinstance(records, list):
        return dict(branches=[], status='unsupported')
    branches = []
    for n, record in enumerate(records):
        if not isinstance(record, dict): continue
        seed = record.get('bootstrap', {})
        grow = record.get('grow')
        pair = seed.get('image_pair') if isinstance(seed, dict) else None
        if not isinstance(pair, list) or len(pair) != 2 or not all(
                isinstance(x, str) for x in pair):
            continue
        b = branch(1, n + 1)
        b['events'] = [event('initial', pair, excerpt='bootstrap.image_pair')]
        b['failed'] = seed.get('decision') != 'Success'
        b['initial_confirmed'] = not b['failed'] and isinstance(
            grow, dict) and isinstance(grow.get('steps'), list)
        b['complete'] = b['initial_confirmed']
        if b['failed']:
            b['warnings'].append('初始对未成功：' + str(seed.get('decision', '结果未知')))
        seen = set(pair)
        for step_index, step in enumerate(
                grow.get('steps', []) if isinstance(grow, dict) else []):
            names = step.get('images') if isinstance(step, dict) else None
            if not isinstance(names, list) or not all(
                    isinstance(x, str) for x in names):
                b['warnings'].append('报告步骤 %d 格式无法识别。' % (step_index + 1))
                b['complete'] = False
                continue
            fresh = [n for n in names if n not in seen]
            if len(fresh) != len(names): b['warnings'].append('报告重复加入记录已去重。')
            if fresh:
                b['events'].append(
                    event('add',
                          fresh,
                          excerpt='grow.steps[%d].images' % step_index))
                seen.update(fresh)
        branches.append(b)
    return dict(branches=branches,
                status='ready' if branches else 'unsupported')


def discover(project):
    project = Path(project).resolve()
    sfm = project / 'opensfm'
    found = set()
    for base in [project / 'log', project / 'logs', sfm / 'log', sfm / 'logs']:
        if base.is_dir():
            found.update(p for p in base.rglob('*') if p.is_file())
    if sfm.is_dir(): found.update(p for p in sfm.iterdir() if p.is_file())
    paths = []
    for p in found:
        if re.search(r'\.(?:log(?:\.\d+)?|txt)$', p.name,
                     re.I) and p.name != 'image_list.txt':
            resolved = p.resolve()
            if project in resolved.parents: paths.append((resolved, 'log'))
    report = sfm / 'reports/reconstruction.json'
    if report.is_file() and project in report.resolve().parents:
        paths.append((report.resolve(), 'report'))
    return sorted(set(paths), key=lambda x: str(x[0]))


_PARSED_CACHE = OrderedDict()
_CACHE_LOCK = Lock()


def parsed(content, kind, known, version):
    # Cache parsed events, not the potentially large raw log bytes.
    key = (hashlib.sha256(content).digest(), kind, known, version)
    with _CACHE_LOCK:
        if key in _PARSED_CACHE:
            _PARSED_CACHE.move_to_end(key)
            return _PARSED_CACHE[key]
    text = content.decode('utf-8-sig', errors='replace')
    result = parse_log(text,
                       set(known)) if kind == 'log' else parse_report(text)
    with _CACHE_LOCK:
        _PARSED_CACHE[key] = result
        _PARSED_CACHE.move_to_end(key)
        while len(_PARSED_CACHE) > 24:
            _PARSED_CACHE.popitem(last=False)
    return result


def catalog(folder, db, component):
    meta = json.loads((folder / 'summary.json').read_text())
    images = {
        normalized(r['name']): r['id']
        for r in db.execute('SELECT id,name FROM images')
    }
    bases = defaultdict(list)
    for name, i in images.items():
        bases[name.rsplit('/', 1)[-1]].append(i)
    memberships = defaultdict(set)
    for c, i in db.execute('SELECT component,image FROM shots'):
        memberships[i].add(c)
    current = {i for i, cs in memberships.items() if component in cs}
    if not db.execute(
            'SELECT 1 FROM points WHERE component=? UNION SELECT 1 FROM shots WHERE component=? LIMIT 1',
        (component, component)).fetchone():
        abort(404, description='分量不存在')
    point_count = db.execute('SELECT COUNT(*) FROM points WHERE component=?',
                             (component, )).fetchone()[0]
    sources = []
    entries = []
    for path, kind in discover(meta['project']):
        relative = str(path.relative_to(Path(meta['project']).resolve()))
        source_id = hashlib.sha256(relative.encode()).hexdigest()[:20]
        source = dict(source_id=source_id,
                      path=relative,
                      kind=kind,
                      status='ready')
        try:
            before = path.stat()
            if before.st_size > 128 * 1024 * 1024:
                raise ValueError('日志超过 128 MiB，请提供单次运行的文本日志。')
            content = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size,
                                                        after.st_mtime_ns):
                raise ValueError('来源正在变化，请重新扫描。')
            digest = hashlib.sha256(content).hexdigest()
            source.update(fingerprint=digest, modified=after.st_mtime)
            result = parsed(content, kind, tuple(sorted(images)),
                            PARSER_VERSION)
            source['status'] = result['status']
            for original in result['branches']:
                b = json.loads(json.dumps(original))
                b.update(source_id=source_id,
                         path=relative,
                         source_kind=kind,
                         fingerprint=digest,
                         modified=after.st_mtime,
                         cross_checked=False)
                b['key'] = source_id + ':' + b['run_id'] + ':' + b['branch_id']
                entries.append(b)
        except (OSError, ValueError, TypeError, KeyError) as error:
            source.update(status='unreadable', error=str(error))
        sources.append(source)

    def signature(b):
        return [sorted(normalized(n) for n in e['names']) for e in b['events']]

    # Do not merge independent files. Corroborate only identical complete sequences.
    reports = [
        b for b in entries if b['source_kind'] == 'report'
        and b['initial_confirmed'] and b['complete']
    ]
    for b in entries:
        if b['source_kind'] == 'log' and not b['failed'] and b['complete']:
            matches = [r for r in reports if signature(r) == signature(b)]
            if matches:
                b['initial_confirmed'] = True
                b['cross_checked'] = True
                for r in matches:
                    r['cross_checked'] = True
    for b in entries:
        mapped = set()
        ambiguous = []
        unknown = []
        components = set()
        seen = set()
        count = 0
        for e in b['events']:
            e['images'] = []
            for name in e['names']:
                norm = normalized(name)
                relative = norm.split(
                    '/images/', 1
                )[-1] if '/images/' in norm else norm.removeprefix('images/')
                exact = norm if norm in images else relative
                possible = [images[exact]] if exact in images else bases.get(
                    norm.rsplit('/', 1)[-1], [])
                i = possible[0] if len(possible) == 1 else None
                if len(possible) > 1: ambiguous.append(name)
                elif i is None: unknown.append(name)
                if i is not None:
                    mapped.add(i)
                    components.update(memberships[i])
                e['images'].append(
                    dict(name=name,
                         id=i,
                         visible=i in current if i is not None else False))
            # Failed/unconfirmed bootstrap attempts are not successful camera events.
            if e['kind'] == 'initial' and not b['initial_confirmed']:
                e['successful'] = False
            else:
                e['successful'] = not b['failed']
            fresh = set(e['names']) - seen if e['successful'] else set()
            e['ordinal_start'] = count + 1 if b[
                'initial_confirmed'] and fresh else None
            count += len(fresh)
            seen.update(fresh)
            e['ordinal_end'] = count if b[
                'initial_confirmed'] and fresh else None
            e['cumulative'] = count
        b.update(mapped_count=len(mapped),
                 coverage=len(mapped & current),
                 registered_images=len(current),
                 unknown_names=unknown,
                 ambiguous_names=ambiguous,
                 sequence_complete=b['complete'] and b['initial_confirmed'],
                 playable=not b['failed'] and not ambiguous
                 and len(components) <= 1 and bool(mapped & current)
                 and any(e['successful'] for e in b['events']))
        if ambiguous: b['warnings'].append('图像名称对应存在歧义，不能自动进行三维回放。')
        if len(components) > 1: b['warnings'].append('该分支对应多个最终分量，不能自动进行三维回放。')
        if unknown:
            b['warnings'].append('%d 张图像无法对应当前项目，事件保留但不显示相机。' % len(unknown))
        if not b['complete']: b['warnings'].append('缺少完整结束记录，仅回放已记录片段。')
        if not b['initial_confirmed'] and not b['failed']:
            b['warnings'].append('初始对未确认，不显示完整加入序号。')
        if b['coverage'] < len(current):
            b['warnings'].append('本分支覆盖当前分量 %d/%d 张图像。' %
                                 (b['coverage'], len(current)))
        if b['summary_points'] is not None and b[
                'summary_points'] != point_count:
            b['warnings'].append('日志结束点数 %d 与当前模型 %d 不同；历史顺序叠加到当前最终模型。' %
                                 (b['summary_points'], point_count))
        b['time'] = next(
            (e['timestamp'] for e in b['events'] if e['timestamp']), None)
        b['event_count'] = sum(e['successful'] for e in b['events'])
        b['image_count'] = count
    entries.sort(key=lambda b: (b['playable'], b['initial_confirmed'], b[
        'coverage'], b['sequence_complete'], b['source_kind'] == 'log', b[
            'time'] or '', b['modified']),
                 reverse=True)
    return dict(sources=sources,
                entries=entries,
                recommended=next((b['key'] for b in entries if b['playable']),
                                 None))


def register_replay_api(app):

    @app.get('/api/replay/sources')
    def replay_sources():
        result = catalog(g.folder, g.db, int(request.args.get('component', 0)))
        for b in result['entries']:
            b['initial_pair'] = b['events'][0]['names'] if b['events'] and b[
                'events'][0]['kind'] == 'initial' else []
            del b['events']
        return jsonify(result)

    @app.get('/api/replay/events')
    def replay_events():
        result = catalog(g.folder, g.db, int(request.args.get('component', 0)))
        b = next((b for b in result['entries']
                  if all(b[k] == request.args.get(k)
                         for k in ('source_id', 'run_id', 'branch_id'))), None)
        if b is None: abort(404, description='过程来源不存在，请重新扫描。')
        expected = request.args.get('fingerprint')
        if expected and expected != b['fingerprint']:
            abort(409, description='过程来源已变化，请重新扫描并加载回放。')
        return jsonify(b)
