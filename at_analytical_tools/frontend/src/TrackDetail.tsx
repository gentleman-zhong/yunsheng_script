import {useEffect} from 'react';
import {Observation,Track,fmt,url,useLoad,fateName} from './api';
import {xy} from './imageUtils';

export function ObservationCards({observations,onObservation}:{observations:Observation[];onObservation:(o:Observation)=>void}){
 return <div className="observations">{observations.map(o=>{
  const pos=xy([o.x,o.y],o.width,o.height),error=o.retained?o.error:o.reference_error;
  return <button className={'observation fate-'+o.fate} key={`${o.track}:${o.image}:${o.feature}:${o.retained}`} onClick={()=>onObservation(o)}>
   <div className="observation-image" style={{aspectRatio:`${o.width||16}/${o.height||9}`}}><img loading="lazy" src={url(`/api/images/${o.image}/asset?size=thumb`)}/>{o.width>0&&o.height>0&&Number.isFinite(pos[0]+pos[1])&&<i style={{left:pos[0]/o.width*100+'%',top:pos[1]/o.height*100+'%'}}/>}</div>
   <strong title={o.name}>{o.name.split('/').pop()}</strong><span>{fateName(o.fate)}</span><span>特征 #{o.feature}{o.coordinate_changed?' · 坐标变化':''}</span>
   <small>{o.retained?'最终误差':'最终几何参考残差'}：{fmt(error,3)} px</small>
  </button>
 })}</div>;
}
export default function TrackDetail({id,component,onObservation,onPoint,onCameras}:{id:string;component:number;onObservation:(o:Observation)=>void;onPoint:(p:number[]|null)=>void;onCameras:(ids:number[])=>void}){
 const {data,error,loading}=useLoad<Track>(`/api/tracks/${encodeURIComponent(id)}?component=${component}`);
 useEffect(()=>{const p=data?.points.find(p=>p.component===component);onPoint(p?[p.x,p.y,p.z]:null);onCameras(data?.observations.filter(o=>o.retained&&o.fate!=='ambiguous').map(o=>o.image)||[])},[data,component]);
 if(!data)return <div className="panel-pad">{error|| (loading?'加载 Track…':'')}</div>;
 const change=data.component_change,final=data.observations.filter(o=>o.retained&&o.fate!=='ambiguous'),reference=data.observations.filter(o=>!o.retained||o.fate==='ambiguous');
 return <div className="track-detail"><span className="eyebrow">TRACK / 两阶段对照</span><h2>Track {id}</h2>
 <p>全项目观测：{fmt(data.length)} → {fmt(data.final_length)}</p>
 {change&&<div className="notice">当前分量：{fateName(change.status)} · {fmt(change.raw_length)} → {fmt(change.final_length)} 次观测</div>}
 {data.source_status!=='ready'&&<p className="inline-error">最终 Tracks 不可用，不能判断保留与删除。</p>}
 {data.points.filter(p=>p.component===component).map(p=><div className="coordinate" key={p.idx}>最终三维点：{[p.x,p.y,p.z].map(v=>fmt(v,3)).join(' / ')}</div>)}
 {!data.points.some(p=>p.component===component)&&<p className="hint">当前分量没有最终三维点，不能计算重投影误差。</p>}
 <h3>最终保留观测 · {final.length}</h3><ObservationCards observations={final} onObservation={onObservation}/>
 <details className="reference-observations"><summary>输入观测对照 · {reference.length}</summary><p className="hint">下列观测未纳入最终质量统计。可计算值是相对于最终几何的参考残差；未保留不等于已证实误匹配。</p><ObservationCards observations={reference} onObservation={onObservation}/></details>
 </div>;
}
