"""Per-image feature cohorts; stage rates always use the previous stage."""
def feature_stages(observations, feature_count, input_status, final_status):
    raw = [o for o in observations if o['input_track']]
    raw_ids = {o['feature'] for o in raw}
    outside = {i for i in raw_ids if i < 0 or (feature_count is not None and i >= feature_count)}
    cohort = raw_ids - outside
    point_ids = {o['feature'] for o in raw if o['idx'] is not None} & cohort
    kept_ids = {o['feature'] for o in raw if o['retained'] and o['idx'] is not None and o['fate'] not in ('ambiguous','unknown')} & point_ids
    ambiguous = any(o.get('ambiguous') or o['fate']=='ambiguous' for o in observations)
    ready = input_status == 'ready'
    final_ready = ready and final_status == 'ready' and not ambiguous
    counts = [feature_count, len(cohort) if ready else None, len(point_ids) if ready else None, len(kept_ids) if final_ready else None]
    rates = [counts[i]/counts[i-1] if counts[i] is not None and counts[i-1] not in (None,0) else None for i in range(1,4)]
    final_only = {o['feature'] for o in observations if not o['input_track'] and o['retained'] and o['idx'] is not None and o['fate']!='ambiguous'}
    return dict(counts=counts, rates=rates, overall=counts[3]/counts[0] if counts[3] is not None and counts[0] not in (None,0) else None,
                outside_feature_ids=len(outside), final_added_features=len(final_only-kept_ids), ambiguous=ambiguous,
                input_status=input_status, final_status=final_status, cohort='input_feature_ids_current_component')
