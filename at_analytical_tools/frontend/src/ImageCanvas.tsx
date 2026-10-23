import {useEffect,useMemo,useRef,useState} from 'react';
import {FeatureData,Observation,url,generation,fmt,fateName,percent} from './api';
import {FeatureFilter,featureFilters,featureColors,featureLabels,featureKind,matchesFeature,groupFeatures} from './featureTypes';
import {Dialog,fullscreen} from './Dialog';
import {useImage,xy} from './imageUtils';

type View={cx:number;cy:number;scale:number|null};
const savedViews=new Map<string,View>();
interface Props{id:number;data:FeatureData;selectedFeature?:number|null;target?:Observation|null;onFeature:(id:number,obs:Observation|undefined)=>void;threshold?:number;name?:string;openToken?:number;onTrack?:(id:string)=>void;onPoint?:(idx:number)=>void;onCloseExpanded?:()=>void}
export default function ImageCanvas({id,data,selectedFeature,target,onFeature,threshold=3,name,openToken=0,onTrack,onPoint,onCloseExpanded}:Props){
 const key=generation+':'+id;
 const initial=()=>savedViews.get(key)||{cx:data.width/2,cy:data.height/2,scale:null};
 const [view,setView]=useState<View>(initial),[expanded,setExpanded]=useState(false),[original,setOriginal]=useState(false),[show,setShow]=useState(true),[errors,setErrors]=useState(false),[filter,setFilter]=useState<FeatureFilter>('all'),[radius,setRadius]=useState(1.2),[opacity,setOpacity]=useState(.8),[dragging,setDragging]=useState(false);
 const [size,setSize]=useState({w:400,h:300});const canvas=useRef<HTMLCanvasElement>(null),host=useRef<HTMLDivElement>(null),frame=useRef<HTMLDivElement>(null);
 const drag=useRef<{x:number;y:number;view:View;scale:number;moved:boolean}|null>(null);
 const {image,error}=useImage(url(`/api/images/${id}/asset?size=${original||expanded?'original':'preview'}`));
 const fit=Math.min(size.w/(data.width||1),size.h/(data.height||1)),scale=view.scale??fit;
 const offset={x:size.w/2-view.cx*scale,y:size.h/2-view.cy*scale};
 const groups=useMemo(()=>groupFeatures(data.observations),[data.observations]);
 const obs=useMemo(()=>{
  const result=new Map([...data.observations].sort((a,b)=>Number(a.retained)-Number(b.retained)).map(o=>[o.feature,o]));
  if(target&&target.image===id&&target.feature===selectedFeature)result.set(target.feature,target);
  return result;
 },[data.observations,target,id,selectedFeature]);
 const featurePoints=useMemo(()=>{
  const result=new Map(data.points.map((p,i)=>[i,p]));
  for(const o of obs.values())result.set(o.feature,[o.x,o.y]);
  return result;
 },[data.points,obs]);
 const pointAt=(i:number)=>featurePoints.get(i)||[null,null];
 const filterCounts=useMemo(()=>Object.fromEntries(featureFilters.map(([key])=>[key,[...featurePoints.keys()].filter(i=>matchesFeature(groups.get(i)||[],key)).length])),[featurePoints,groups]);
 const visible=(i:number)=>matchesFeature(groups.get(i)||[],filter);
 const featureKinds=useMemo(()=>new Map([...featurePoints.keys()].map(i=>[i,featureKind(groups.get(i)||[],threshold,data.input_status==='ready')])),[featurePoints,groups,threshold,data.input_status]);
 const legendCounts=useMemo(()=>{
  const counts=new Map<keyof typeof featureColors,number>();
  for(const [i,kind] of featureKinds){if(matchesFeature(groups.get(i)||[],filter))counts.set(kind,(counts.get(kind)||0)+1)}
  return counts;
 },[featureKinds,groups,filter]);
 const stages=data.stages;
 useEffect(()=>{setView(initial())},[key]);
 useEffect(()=>{savedViews.set(key,view);if(savedViews.size>100)savedViews.delete(savedViews.keys().next().value!)},[view,key]);
 useEffect(()=>{if(openToken){setExpanded(true);setErrors(true);setFilter('all');if(selectedFeature!=null){const [cx,cy]=xy(pointAt(selectedFeature),data.width,data.height);if(Number.isFinite(cx+cy))setView({cx,cy,scale:1})}}},[openToken]);
 useEffect(()=>{const el=host.current;if(!el)return;const r=new ResizeObserver(()=>setSize({w:el.clientWidth,h:el.clientHeight}));r.observe(el);return()=>r.disconnect()},[expanded]);

 function zoom(factor:number,px=size.w/2,py=size.h/2){setView(v=>{const current=v.scale??fit,next=Math.max(fit*.25,Math.min(16,current*factor));return {cx:v.cx+(px-size.w/2)*(1/current-1/next),cy:v.cy+(py-size.h/2)*(1/current-1/next),scale:next}})}
 useEffect(()=>{const c=canvas.current;if(!c)return;const wheel=(e:WheelEvent)=>{e.preventDefault();const r=c.getBoundingClientRect();zoom(Math.exp(-Math.max(-100,Math.min(100,e.deltaY))*.002),e.clientX-r.left,e.clientY-r.top)};c.addEventListener('wheel',wheel,{passive:false});return()=>c.removeEventListener('wheel',wheel)},[size,expanded,fit]);
 useEffect(()=>{const c=canvas.current;if(!c)return;const dpr=Math.min(devicePixelRatio,2);c.width=size.w*dpr;c.height=size.h*dpr;const ctx=c.getContext('2d')!;ctx.scale(dpr,dpr);ctx.fillStyle='#0b1419';ctx.fillRect(0,0,size.w,size.h);
 if(image)ctx.drawImage(image,offset.x,offset.y,data.width*scale,data.height*scale);
 if(show)featurePoints.forEach((point,i)=>{if(!visible(i))return;const [u,v]=xy(point,data.width,data.height),x=u*scale+offset.x,y=v*scale+offset.y;if(!Number.isFinite(x+y)||x<0||x>size.w||y<0||y>size.h)return;
 const o=obs.get(i),selected=i===selectedFeature;ctx.globalAlpha=selected?1:opacity;ctx.fillStyle=featureColors[featureKinds.get(i)!];ctx.beginPath();ctx.arc(x,y,selected?5:radius,0,Math.PI*2);ctx.fill();
 if(selected){ctx.globalAlpha=1;ctx.strokeStyle='#17252a';ctx.lineWidth=2;ctx.stroke();ctx.strokeStyle='#fff';ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.stroke();
 const px=o?.retained?o.px:o?.reference_px,py=o?.retained?o.py:o?.reference_py;
 if(errors&&px!=null&&py!=null){const [pu,pv]=xy([px,py],data.width,data.height),ex=pu*scale+offset.x,ey=pv*scale+offset.y;ctx.save();ctx.beginPath();ctx.rect(0,0,size.w,size.h);ctx.clip();if(Number.isFinite(ex+ey)){ctx.strokeStyle='#ff9a75';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);ctx.moveTo(ex-6,ey);ctx.lineTo(ex+6,ey);ctx.moveTo(ex,ey-6);ctx.lineTo(ex,ey+6);ctx.stroke()}ctx.restore()}}
 });ctx.globalAlpha=1;
 },[image,size,view,data,show,errors,filter,selectedFeature,threshold,opacity,radius,expanded,target,featureKinds]);
 const selected=selectedFeature==null?undefined:obs.get(selectedFeature),coords=selectedFeature==null?null:xy(pointAt(selectedFeature),data.width,data.height);
 const closeExpanded=()=>{setExpanded(false);onCloseExpanded?.()};
 const content=<div className={'image-inspector '+(expanded?'expanded-image':'')} ref={frame}>
 {expanded&&<header className="viewer-heading"><div><strong>图像特征</strong><span title={name}>{name||`图像 #${id}`}</span></div><button onClick={()=>fullscreen(frame.current)}>浏览器全屏</button><button aria-label="关闭大图" onClick={closeExpanded}>关闭 ×</button></header>}
 <div className="canvas-options"><label><input type="checkbox" checked={show} onChange={e=>setShow(e.target.checked)}/>特征点</label><label><input type="checkbox" checked={errors} onChange={e=>setErrors(e.target.checked)}/>选中点投影线</label><select aria-label="特征类型" value={filter} onChange={e=>setFilter(e.target.value as FeatureFilter)}>{featureFilters.map(([key,label])=><option key={key} value={key}>{label} · {fmt(filterCounts[key])}</option>)}</select>
 {expanded&&<><label>点大小<input aria-label="特征点大小" type="range" min=".5" max="4" step=".1" value={radius} onChange={e=>setRadius(+e.target.value)}/></label><label>透明度<input aria-label="特征点透明度" type="range" min=".1" max="1" step=".1" value={opacity} onChange={e=>setOpacity(+e.target.value)}/></label></>}</div>
 {expanded?<div className="feature-stage-panel"><p className="feature-filter-hint">{featureFilters.find(([key])=>key===filter)?.[2]}</p>
 <div className="feature-legend" aria-label="特征颜色图例">{Object.entries(featureColors).map(([key,color])=><span key={key}><i style={{background:color}}/>{featureLabels[key as keyof typeof featureLabels]} <b>{fmt(legendCounts.get(key as keyof typeof featureColors)||0)}</b></span>)}</div>
 <p className="hint">高误差：≥ {threshold} px；白色外圈表示选中，中心保留状态颜色。图例数量按当前筛选统计。</p>
 {stages?<details className="feature-stages" open={expanded||undefined}><summary>特征存活率 · 输入观测最终保留 / 全部特征：{percent(stages.overall)}</summary><div className="feature-stage-grid">{['全部提取特征','形成输入 Tracks','输入 Tracks 形成三维点','其中本图最终保留'].map((label,index)=><div key={label}><span>{label}</span><strong>{fmt(stages.counts[index])}</strong>{index>0&&<><small>相对上一阶段：<b>{percent(stages.rates[index-1])}</b></small><small>占全部特征：{percent(stages.counts[index]!=null&&stages.counts[0]?stages.counts[index]!/stages.counts[0]!:null)}</small></>}</div>)}</div>
 <p className="hint">按本图特征 ID 去重，在当前重建分量内追踪同一批输入观测。每步转化率 = 当前阶段 / 上一阶段；高误差仍计入最终保留。分母为零或数据不足时显示“—”。</p>
 {data.resource_error&&<p className="hint">特征文件不可用：全部特征数量及以其为分母的转化率不可计算。</p>}
 {stages.input_status!=='ready'&&<p className="inline-error">输入 Tracks 不可用，不能计算输入链路的存活率。</p>}
 {(stages.final_status!=='ready'||stages.ambiguous)&&<p className="inline-error">最终观测缺失或归属存在歧义，最终保留率不可计算。</p>}
 {stages.outside_feature_ids>0&&<p className="hint">{stages.outside_feature_ids} 个输入特征 ID 无法对应特征文件，已排除转化率计算；其 Tracks 坐标仍可查看。</p>}
 {stages.final_added_features>0&&<p className="hint">另有 {stages.final_added_features} 个特征通过最终新增关联被保留，展示在三维点筛选中，不计入原输入观测的逐级存活率。</p>}
 </details>:<p className="hint">请重启后端获取逐阶段特征存活率。</p>}</div>:<div className="feature-overview">
   <div className="feature-overview-metrics">
     <div><span>形成输入 Tracks</span><strong>{fmt(stages?.counts[1])}</strong></div>
     <div><span>输入特征最终保留</span><strong>{fmt(stages?.counts[3])}</strong></div>
     <div className="feature-overview-rate" title="输入特征最终保留 / 全部提取特征；高误差观测仍计入保留"><span>特征存活率</span><strong>{percent(stages?.overall)}</strong></div>
   </div>
   <div className="feature-overview-footer"><span>当前分量 · 存活率以全部提取特征为分母</span><button onClick={()=>setExpanded(true)}>查看详细图像特征 →</button></div>
   {(!stages||stages.overall==null)&&<p className="hint">数据不足，存活率暂不可计算；详情中可查看原因。</p>}
   <div className="feature-legend feature-legend-compact" aria-label="特征颜色图例">{Object.entries(featureColors).filter(([key])=>(legendCounts.get(key as keyof typeof featureColors)||0)>0).map(([key,color])=><span key={key} title={featureLabels[key as keyof typeof featureLabels]}><i style={{background:color}}/>{({retained:'本图保留',high:'高误差',removed:'观测未保留',absent:'未形成三维点',none:'无 Track',unknown:'状态未定'} as Record<string,string>)[key]}</span>)}</div>
 </div>}
 <div className="image-actions"><button aria-label="放大图像" onClick={()=>zoom(1.3)}>＋</button><button aria-label="缩小图像" onClick={()=>zoom(1/1.3)}>−</button><button onClick={()=>setView({cx:data.width/2,cy:data.height/2,scale:null})}>适应窗口</button><button onClick={()=>{setOriginal(true);setView(v=>({...v,scale:1}))}}>原图 100%</button><span className="zoom-value">{Math.round(scale*100)}%</span></div>
 <div ref={host} className="image-canvas-wrap"><canvas ref={canvas} aria-label="二维特征图" data-scale={scale} data-center={`${view.cx},${view.cy}`} style={{cursor:dragging?'grabbing':'grab'}} onPointerDown={e=>{if(e.button!==0)return;e.currentTarget.setPointerCapture(e.pointerId);drag.current={x:e.clientX,y:e.clientY,view,scale,moved:false}}} onPointerMove={e=>{const d=drag.current;if(!d)return;const dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.hypot(dx,dy)>5)d.moved=true;if(d.moved){setDragging(true);setView({...d.view,cx:Math.max(0,Math.min(data.width,d.view.cx-dx/d.scale)),cy:Math.max(0,Math.min(data.height,d.view.cy-dy/d.scale))})}}} onPointerUp={e=>{const d=drag.current;drag.current=null;setDragging(false);if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);if(!d||d.moved||!show)return;const r=e.currentTarget.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,distance=9;featurePoints.forEach((p,i)=>{if(!visible(i))return;const [u,v]=xy(p,data.width,data.height),dist=Math.hypot(u*scale+offset.x-mx,v*scale+offset.y-my);if(dist<distance){best=i;distance=dist}});if(best>=0)onFeature(best,obs.get(best))}} onPointerCancel={()=>{drag.current=null;setDragging(false)}}/>
 {!image&&<div className="canvas-message">{error||'加载图像…'}</div>}
 </div>

 {data.resource_error&&<p className="hint">{data.resource_error}；仍可查看 Tracks 自带观测位置。</p>}

 {selectedFeature!=null&&!visible(selectedFeature)&&<p className="hint panel-pad">当前选中特征不在此筛选中。<button onClick={()=>setFilter('all')}>显示全部特征</button></p>}
 {selectedFeature!=null&&<div className="feature-info"><strong>特征 #{selectedFeature}</strong><span>{coords?.map(v=>fmt(v,1)).join(' / ')} px</span>{selected?.error!=null&&<span>最终误差 {fmt(selected.error,3)} px</span>}{selected&&<span>{fateName(selected.fate)}{!selected.retained&&selected.reference_error!=null?` · 最终几何参考残差 ${fmt(selected.reference_error,3)} px`:''}</span>}{selected?.track&&onTrack&&<button onClick={()=>onTrack(selected.track)}>查看 Track</button>}{selected?.idx!=null&&onPoint&&<button onClick={()=>onPoint(selected.idx!)}>定位三维点</button>}</div>}
 {expanded&&<footer className="viewer-help">滚轮缩放 · 拖动平移 · 点击选点 · Esc 返回　|　{data.width} × {data.height} · 原分辨率</footer>}
 </div>;
 return expanded?<Dialog title="图像特征大图" onClose={closeExpanded}>{content}</Dialog>:content;
}
