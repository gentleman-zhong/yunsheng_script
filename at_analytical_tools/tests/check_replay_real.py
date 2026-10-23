"""Verify log/report sequence against a read-only viewer index and source hashes."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from server import create_app
from replay import discover


def check(folder):
    meta=json.loads((folder/'summary.json').read_text())
    inputs=[p for p,_ in discover(meta['project'])]+[Path(meta['reconstruction'])]
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    before=hashes()
    client=create_app(SimpleNamespace(active=folder,status=lambda:{'available':True})).test_client()
    response=client.get('/api/replay/sources?component=0');assert response.status_code==200,response.json
    result=response.json
    entry=next(b for b in result['entries'] if b['key']==result['recommended'])
    detail=client.get('/api/replay/events',query_string={k:entry[k] for k in ('source_id','run_id','branch_id','fingerprint')})
    assert detail.status_code==200,detail.json
    timeline=detail.json
    assert entry['source_kind']=='log' and timeline['playable'] and timeline['sequence_complete']
    text=(Path(meta['project'])/entry['path']).read_text()
    pair=re.search(r'Starting reconstruction with (.+?) and (.+)',text).groups()
    added=re.findall(r'Adding (.+?) to the reconstruction',text)
    expected=list(pair)+added
    actual=[name for e in timeline['events'] for name in e['names']]
    assert actual==expected
    assert len(expected)==len(set(expected))==166
    assert len(timeline['events'])==165 and timeline['coverage']==166 and timeline['cross_checked']
    assert expected[0].endswith('_0099_V.JPG') and expected[1].endswith('_0100_V.JPG')
    assert expected[2].endswith('_0101_V.JPG') and expected[3].endswith('_0097_V.JPG')
    report=next(b for b in result['entries'] if b['source_kind']=='report')
    report_result=client.get('/api/replay/events',query_string={k:report[k] for k in ('source_id','run_id','branch_id','fingerprint')}).json
    assert [e['names'] for e in timeline['events']]==[e['names'] for e in report_result['events']]
    assert before==hashes()
    return dict(source_unchanged=True,initial_pair=list(pair),third=expected[2],fourth=expected[3],
                camera_count=len(actual),event_count=len(timeline['events']),log_report_order_equal=True,
                warnings=timeline['warnings'],source=entry['path'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('index',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=check(args.index.resolve())
    rendered=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(rendered+'\n')
    print(rendered)
