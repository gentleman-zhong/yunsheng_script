import type {Observation} from './api';
export type FeatureFilter='all'|'input'|'reconstructed'|'retained';
export const featureFilters:[FeatureFilter,string,string][]=[
 ['all','全部特征','展示全部提取特征及 Tracks 自带的观测位置，按重建状态着色。'],
 ['input','形成输入 Tracks 的特征','包括最终保留、未形成三维点，以及三维点存在但本图观测未保留的输入特征。'],
 ['reconstructed','对应已重建三维点','当前分量中存在对应三维点，包括本图观测未保留的特征。'],
 ['retained','该图最终保留的三维点','当前图像中被最终重建保留的观测，包含高误差观测。'],
];
export const inputObservation=(o:Observation)=>o.input_track??o.fate!=='added';
export const retainedObservation=(o:Observation)=>o.retained&&o.idx!=null&&!['ambiguous','unknown'].includes(o.fate);
export function matchesFeature(rows:Observation[],filter:FeatureFilter){return filter==='all'||rows.some(o=>filter==='input'?inputObservation(o):filter==='reconstructed'?o.idx!=null:retainedObservation(o))}
export const featureColors={retained:'#45e69c',high:'#ff5757',removed:'#ffbd40',absent:'#c88bff',none:'#5da7ff',unknown:'#a5b0b8'};
export const featureLabels={retained:'本图最终保留',high:'最终保留 · 高误差',removed:'三维点在 · 本图观测未保留',absent:'输入 Track 无三维点',none:'无 Track 记录',unknown:'状态未确定'};
export function featureKind(rows:Observation[],threshold:number,inputAvailable=true):keyof typeof featureColors{
 if(rows.some(o=>retainedObservation(o)&&o.error!=null&&o.error>=threshold))return 'high';
 if(rows.some(retainedObservation))return 'retained';
 if(!rows.length)return inputAvailable?'none':'unknown';
 if(rows.some(o=>o.fate==='ambiguous'||o.fate==='unknown'))return 'unknown';
 if(rows.some(o=>o.idx!=null))return 'removed';
 if(rows.some(o=>inputObservation(o)&&o.fate==='no_final_point'))return 'absent';
 return 'unknown';
}
export function groupFeatures(observations:Observation[]){const map=new Map<number,Observation[]>();for(const o of observations){const rows=map.get(o.feature)||[];rows.push(o);map.set(o.feature,rows)}return map}
