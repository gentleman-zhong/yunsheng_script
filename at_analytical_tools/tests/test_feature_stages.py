import pytest
from feature_stages import feature_stages
from test_comparison import project, open_client, final_subset
from test_viewer import dataset


def observation(feature,point=None,retained=False,input_track=True,fate=None):
    return dict(feature=feature,idx=point,retained=retained,input_track=input_track,
                fate=fate or ('retained' if retained else 'removed_observation' if point is not None else 'no_final_point'))


def test_sequential_survival_and_feature_deduplication():
    rows=[observation(i) for i in range(20)]
    rows+=[observation(i,point=i,retained=i<2) for i in range(4)]
    rows+=[observation(0,point=0,retained=True)]
    result=feature_stages(rows,100,'ready','ready')
    assert result['counts']==[100,20,4,2]
    assert result['rates']==[.2,.2,.5]
    assert result['overall']==.02


def test_added_invalid_missing_and_zero_denominators():
    rows=[observation(0,point=0,retained=True),observation(1,point=1),observation(2),
          observation(9,point=9,retained=True,input_track=False,fate='added'),observation(50,point=50,retained=True)]
    result=feature_stages(rows,10,'ready','ready')
    assert result['counts']==[10,3,2,1]
    assert result['outside_feature_ids']==1 and result['final_added_features']==1
    assert feature_stages(rows,None,'ready','ready')['rates'][0] is None
    assert feature_stages(rows,10,'missing','ready')['counts']==[10,None,None,None]
    assert feature_stages(rows,10,'ready','missing')['counts']==[10,3,2,None]
    assert feature_stages([],0,'ready','ready')['rates']==[None,None,None]
    rows.append(observation(1,point=1,retained=True,fate='ambiguous'))
    assert feature_stages(rows,10,'ready','ready')['counts'][3] is None


def test_feature_api_stage_counts_and_provenance(project,tmp_path):
    final_subset(project)
    client,out=open_client(project,tmp_path)
    retained=client.get('/api/images/0/features').json
    discarded=client.get('/api/images/2/features').json
    assert retained['stages']['counts']==[3,2,1,1]
    assert retained['stages']['rates']==pytest.approx([2/3,1/2,1])
    assert discarded['stages']['counts']==[3,2,1,0]
    assert all(o['input_track'] for o in discarded['observations'])
