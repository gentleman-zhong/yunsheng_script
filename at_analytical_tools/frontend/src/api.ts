import {useEffect, useState} from 'react';
export let generation = '';
export function setGeneration(value:string){generation=value;}
export function url(path:string){return path+(path.includes('?')?'&':'?')+'generation='+encodeURIComponent(generation);}
export async function get<T>(path:string, signal?:AbortSignal):Promise<T>{
 const r=await fetch(url(path),{signal}); if(!r.ok){let message=r.statusText;try{message=(await r.json()).error||message;}catch{}throw new Error(message);}return r.json();
}
interface LoadState<T> {
 path: string;
 gen: string;
 data: T | null;
 error: string;
 loading: boolean;
}

export function useLoad<T>(path: string | null) {
 const [state, setState] = useState<LoadState<T> | null>(null);
 const gen = generation;
 useEffect(() => {
  if (!path) return;
  const controller = new AbortController();
  const pending = {path, gen, data: null, error: '', loading: true};
  setState(pending);
  get<T>(path, controller.signal).then(data => {
   if (!controller.signal.aborted) setState({...pending, data, loading: false});
  }).catch(error => {
   if (!controller.signal.aborted) setState({...pending, error: String(error instanceof Error ? error.message : error), loading: false});
  });
  return () => controller.abort();
 }, [path, gen]);
 // Never expose data, errors or loading state belonging to another request.
 if (!path) return {data: null, error: '', loading: false};
 if (state?.path !== path || state.gen !== gen) return {data: null, error: '', loading: true};
 return {data: state.data, error: state.error, loading: state.loading};
}
export const fmt=(n:number|null|undefined,d=0)=>n==null||!Number.isFinite(n)?'—':Math.abs(n)>=1e8?n.toExponential(3):n.toLocaleString('zh-CN',{maximumFractionDigits:d});
export const short=(s:string)=>s.split('/').pop()?.replace(/^\d+_/,'').slice(0,24)||s;
export interface Camera{id:number;name:string;center:number[];world_center:number[];rotation:number[][];rays:number[][];supported:boolean;model:string}
export interface Scene{id:number;points:number;cameras:Camera[];origin:number[];extent:number;bounds:number[][];coordinates:string}
export interface ImageInfo{id:number;name:string;width:number;height:number;features:number;neighbors:number;matches:number;problem:string;stats?:Stats[];registered?:boolean;max?:number;mean?:number}
export interface Observation{input_track?:boolean;feature:number;track:string;image:number;name:string;x:number;y:number;width:number;height:number;error:number|null;px:number|null;py:number|null;state:string;idx?:number|null;retained:boolean;fate:string;reference_error?:number|null;reference_px?:number|null;reference_py?:number|null;coordinate_changed?:boolean;input_x?:number|null;input_y?:number|null}
export interface FeatureData{points:(number|null)[][];observations:Observation[];width:number;height:number;resource_error?:string|null;source_status?:string;input_status?:string;stages?:FeatureStages}
export interface Track{id:string;length:number;final_length:number|null;change:TrackChange;component_change:TrackChange|null;source_status:string;points:{component:number;idx:number;x:number;y:number;z:number}[];observations:Observation[];stats:{mean:number;max:number;p95:number;median:number}[]}
export interface MatchData{name_a:string;name_b:string;a:number;b:number;count:number;invalid:number;pairs:number[][];a_points:number[][];b_points:number[][];width_a:number;height_a:number;width_b:number;height_b:number;total?:number;offset?:number;tracks?:string[];retained?:boolean[];available?:boolean}
export interface List<T>{items:T[];total:number}
export interface Stats{count:number;mean:number;median:number;min:number;max:number;p90:number;p95:number;rmse:number;points?:number;unavailable?:number;above_threshold?:number;states?:Record<string,number>;source_status?:string;ambiguous_points?:number;observation_errors?:Stats}
export interface Project{name:string;project:string;reconstruction:string;warnings:string[];warning_count:number;stats:{images:number;features:number;tracks:number;observations:number;track_images:number;track_lengths:Stats;histogram:number[][];matched_pairs:number;empty_pairs:number;invalid_matches:number;reconstructed_tracks:number;errors:Stats[];sources:Record<string,TrackSource>;overlap?:OverlapStats[]}}

export interface PointDiagnostic{component:number;idx:number;track:string;x:number;y:number;z:number;mean:number|null;median:number|null;p95:number|null;max:number|null;rmse:number|null;count:number}
export interface PointDetail extends Pick<PointDiagnostic,'component'|'idx'|'track'|'x'|'y'|'z'>{stats:PointDiagnostic;observations:Observation[];reference_observations:Observation[];source_status:string;ambiguous:boolean}

export interface OverlapStats{component:number;metric:string;mean:number|null;registered_images:number;evaluated_images:number;eligible_pairs:number;parameters:{grid_size:number;max_neighbors:number;min_common_tracks:number;max_error_px:number}}
export interface TrackSource {path:string;status:string;tracks:number|null;observations:number|null;images:number|null;integrity?:Record<string,number>}
export interface TrackChange {track:string;raw_length:number;final_length:number|null;retained:number|null;loss:number|null;added:number|null;status:string;coordinate_changes:number}
export interface ImageChange {image:number;name:string;raw_count:number;final_count:number|null;retained:number|null;loss:number|null;added:number|null;mean:number|null;p95:number|null;retention?:number|null;smoothed?:number|null;order?:number;raw_neighbors?:number;final_neighbors?:number|null}
export interface PairChange {a:number;b:number;name_a:string;name_b:string;raw_count:number;final_count:number|null}
export interface ComparisonSummary {available:boolean;scope:string;component:number;sources:Record<string,TrackSource>;raw_tracks:number;final_tracks:number|null;raw_observations:number;final_observations:number|null;retained_observations:number|null;track_retention:number|null;observation_retention:number|null;removed_surviving:number|null;removed_without_point:number|null;added_observations:number|null;coordinate_changes:number;states:Record<string,number>;migrations:{raw:number;final:number;count:number}[];lengths:{length:number;tracks:number;surviving:number|null;raw_observations:number;retained:number|null}[];timeline:ImageChange[];order_label:string;clues:ImageChange[];topology:{threshold:number;matched:{edges:number;components:number;isolated:number};final:{edges:number;components:number;isolated:number}|null;weakened:number|null;vanished:number|null}}
export const fateName=(f:string)=>({retained:'最终保留',added:'最终新增',removed_observation:'点仍在 · 观测未保留',no_final_point:'无最终三维点',outside_component:'当前分量外',ambiguous:'分量归属歧义',unknown:'最终数据不可用',unchanged:'完整保留',shortened:'部分保留',absent:'未进入最终结果',changed:'关联或坐标变化'}[f]||f);
export const percent=(n:number|null|undefined)=>n==null||!Number.isFinite(n)?'—':(n*100).toFixed(1)+'%';

export interface FeatureStages{counts:(number|null)[];rates:(number|null)[];overall:number|null;outside_feature_ids:number;final_added_features:number;ambiguous:boolean;input_status:string;final_status:string;cohort:string}
