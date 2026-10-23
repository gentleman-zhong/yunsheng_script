import json
import os
import sqlite3
from pathlib import Path

from test_comparison import project, open_client
from test_viewer import dataset
from replay import parse_log, parse_report, split_names


def log_body(pair=('sub/a.jpg','sub/b.jpg'), additions=('sub/c.jpg',), complete=True):
    lines=['2026-10-21 10:00:00,000 INFO: Starting incremental reconstruction',
           '2026-10-21 10:00:01,000 INFO: Starting reconstruction with %s and %s'%pair]
    for name in additions:
        lines.extend(['INFO: %s resection inliers: 20 / 30'%name,'INFO: Adding %s to the reconstruction'%name])
    if complete: lines+=['INFO: Reconstruction 0: 3 images, 999 points','INFO: 1 partial reconstructions in total.']
    return '\n'.join(lines)


def report_body():
    return {'reconstructions':[{'bootstrap':{'image_pair':['sub/a.jpg','sub/b.jpg'],'decision':'Success'},'grow':{'steps':[{'images':['sub/c.jpg']}]}}]}


def write_log(project,text,name='test.log'):
    folder=project/'log';folder.mkdir(exist_ok=True)
    p=folder/name;p.write_text(text);return p


def get_detail(client,entry):
    return client.get('/api/replay/events',query_string={k:entry[k] for k in ('source_id','run_id','branch_id','fingerprint')})


def test_parser_failed_seed_retries_resections_duplicates_and_group():
    text='\x1b[39mINFO: Starting incremental reconstruction\nINFO: Starting reconstruction with bad.jpg and bad2.jpg\nINFO: Could not find initial motion\n'
    text+=log_body(additions=('sub/c.jpg and sub/d.jpg','sub/c.jpg and sub/d.jpg'))
    result=parse_log(text,set())
    assert len(result['branches'])==2
    assert result['branches'][0]['failed'] and not result['branches'][0]['initial_confirmed']
    b=result['branches'][1]
    assert b['initial_confirmed'] and b['complete']
    assert len(b['events'])==2
    assert b['events'][1]['names']==['sub/c.jpg','sub/d.jpg']
    assert b['warnings'] and b['events'][0]['timestamp']=='2026-10-21 10:00:01.000'


def test_parser_fragment_unsupported_nonincremental_and_failed_report():
    b=parse_log('INFO: Adding c.jpg to the reconstruction',set())['branches'][0]
    assert not b['initial_confirmed'] and not b['complete']
    assert parse_log('INFO: Starting planar reconstruction\nINFO: Adding x to the reconstruction',set())['status']=='non_incremental'
    assert parse_log('settings: incremental\nhello',set())['status']=='unsupported'
    b=parse_log(log_body(additions=(),complete=True),set())['branches'][0]
    assert not b['initial_confirmed']
    value=report_body();value['reconstructions'][0]['bootstrap']['decision']='Failed'
    assert parse_report(json.dumps(value))['branches'][0]['failed']
    assert split_names('one and two.jpg and three.jpg',{'one and two.jpg','three.jpg'})==['one and two.jpg','three.jpg']


def test_sources_api_recommends_log_checks_report_and_preserves_evidence(project,tmp_path):
    write_log(project,log_body())
    report=project/'opensfm/reports/reconstruction.json';report.parent.mkdir();report.write_text(json.dumps(report_body()))
    client,out=open_client(project,tmp_path)
    response=client.get('/api/replay/sources');assert response.status_code==200
    entries=response.json['entries'];assert len(entries)==2
    assert entries[0]['source_kind']=='log' and entries[0]['key']==response.json['recommended']
    assert all(e['cross_checked'] for e in entries)
    b=get_detail(client,entries[0]).json
    assert b['image_count']==3 and b['coverage']==3 and b['playable']
    assert b['events'][0]['ordinal_start']==1 and b['events'][0]['ordinal_end']==2
    assert b['events'][1]['ordinal_start']==3 and b['events'][1]['cumulative']==3
    assert b['events'][0]['line']==2
    assert any('999' in w for w in b['warnings'])
    assert client.get('/api/replay/events?source_id=../../etc/passwd').status_code==404
    assert client.get('/api/replay/sources?component=999').status_code==404


def test_missing_log_report_fallback_and_changed_same_size_same_time(project,tmp_path):
    report=project/'opensfm/reports/reconstruction.json';report.parent.mkdir();report.write_text(json.dumps(report_body()))
    client,out=open_client(project,tmp_path)
    sources=client.get('/api/replay/sources').json
    b=sources['entries'][0];assert b['source_kind']=='report' and b['playable']
    old=report.stat();report.write_text(report.read_text().replace('sub/c.jpg','sub/x.jpg'));os.utime(report,ns=(old.st_atime_ns,old.st_mtime_ns))
    assert get_detail(client,b).status_code==409
    fresh=client.get('/api/replay/sources').json['entries'][0]
    assert fresh['unknown_names']==['sub/x.jpg']
    report.unlink();assert client.get('/api/replay/sources').json['sources']==[]


def test_ambiguous_basenames_and_cross_component_disable_replay(project,tmp_path):
    write_log(project,log_body(pair=('a.jpg','sub/b.jpg')))
    client,out=open_client(project,tmp_path)
    with sqlite3.connect(out/'index.sqlite') as db:
        db.execute("INSERT INTO images(id,name) VALUES(99,'other/a.jpg')")
    b=client.get('/api/replay/sources').json['entries'][0]
    assert not b['playable'] and b['ambiguous_names']==['a.jpg']
    write_log(project,log_body())
    with sqlite3.connect(out/'index.sqlite') as db:
        db.execute('INSERT INTO shots(component,image) VALUES(1,2)')
    b=client.get('/api/replay/sources').json['entries'][0]
    assert not b['playable'] and any('多个最终分量' in w for w in b['warnings'])


def test_initial_attempt_not_counted_and_partial_numbering(project,tmp_path):
    path=write_log(project,log_body(additions=(),complete=True))
    client,out=open_client(project,tmp_path)
    b=client.get('/api/replay/sources').json['entries'][0]
    assert not b['playable'] and b['image_count']==0
    path.write_text('INFO: Adding sub/c.jpg to the reconstruction')
    b=client.get('/api/replay/sources').json['entries'][0]
    assert b['playable'] and not b['sequence_complete']
    e=get_detail(client,b).json['events'][0]
    assert e['ordinal_start'] is None and e['cumulative']==1


def test_discovery_multiple_runs_and_unsupported_sources(project,tmp_path):
    write_log(project,log_body()+'\n'+log_body(complete=False))
    write_log(project,'INFO: Starting triangulation reconstruction','global.log')
    write_log(project,'not a known format','unknown.txt')
    report=project/'opensfm/reports/reconstruction.json';report.parent.mkdir();report.write_text('{invalid')
    client,out=open_client(project,tmp_path)
    result=client.get('/api/replay/sources').json
    assert len(result['entries'])==2 and result['entries'][0]['sequence_complete']
    assert {b['run_id'] for b in result['entries']}=={'1','2'}
    assert {s['status'] for s in result['sources']}=={'ready','non_incremental','unsupported','unreadable'}
