import {fmt,short,url} from './api';
export interface Relation{id:number;name:string;direct_count:number;in_component:boolean}
export interface Relations{items:Relation[];selected_registered:boolean}
export function visibleRelations(data:Relations|null){return (data?.items||[]).filter(r=>r.in_component&&r.direct_count>0)}
export function RelationControls({data,error,selected}:{data:Relations|null;error:string;selected:boolean}){
 const shown=visibleRelations(data),outside=data?.items.filter(r=>!r.in_component).length||0;
 return <div className="relation-controls">{!selected?<p>选择一个相机，查看与它直接匹配的图像。</p>:<><p><strong>直接匹配</strong> · 全项目 <b>{data?fmt(data.items.length):'—'}</b> 张 · 当前分量 <b>{data?fmt(shown.length):'—'}</b> 张{outside>0&&<> · 分量外 {fmt(outside)} 张（见匹配列表）</>}</p><p className="hint">黄色表示当前分量内有直接匹配记录的相机，至少包含一对有效匹配。</p>{data&&!data.selected_registered&&<p className="hint">所选图像未注册到当前分量；仍可查看匹配图像，分量外图像在列表中单列。</p>}</>}{error&&<p className="inline-error">{error}</p>}</div>
}
export function RelationList({data,items,onPair,error}:{data:Relations|null;items:Relation[];onPair:(id:number)=>void;error:string}){
 const row=(r:Relation)=><div className="relation-row" key={r.id}><img loading="lazy" src={url(`/api/images/${r.id}/asset?size=thumb`)}/><div><strong title={r.name}>{short(r.name)}</strong><small>{fmt(r.direct_count)} 对有效直接匹配</small><button onClick={()=>onPair(r.id)}>查看匹配</button></div></div>;
 const outside=data?.items.filter(r=>!r.in_component)||[],sorted=(rows:Relation[])=>[...rows].sort((a,b)=>b.direct_count-a.direct_count||a.id-b.id);
 return <div className="relation-list"><h3>匹配图像 · {data?fmt(data.items.length):'—'}</h3><p className="hint">按有效匹配数量排序。当前分量中的匹配相机在三维视图中标黄。</p>{error?<p className="inline-error">{error}</p>:!data?<p>加载匹配图像…</p>:<><h4>当前分量 · {fmt(items.length)}</h4>{sorted(items).map(row)}{!items.length&&<p className="empty">当前分量没有匹配相机</p>}{outside.length>0&&<details><summary>分量外匹配图像 · {fmt(outside.length)}</summary><p className="hint">这些图像不在当前三维分量中显示，仍可打开双图匹配。</p>{sorted(outside).map(row)}</details>}</>}</div>
}
