import {useEffect,useRef,useState} from 'react';
import {MatchData,Track,FeatureData,fmt,useLoad,url,get,generation,Observation,fateName} from './api';
import {commonMatchTracks} from './matchTracks';
import {useImage,xy} from './imageUtils';
import {Dialog,fullscreen} from './Dialog';
type View={scale:number|null;cx:number;cy:number};
const defaultAppearance={lineOpacity:.8,lineWidth:1.2,pointOpacity:1,pointSize:6};
let savedAppearance={...defaultAppearance};
const saved=new Map<string,{views:View[];selected:number|null}>();
export default function MatchView({pair,onClose,component=0,comparison=false,scope='component',onObservation}:{pair:[number,number];component?:number;comparison?:boolean;scope?:string;onObservation?:(o:Observation)=>void;onClose:()=>void}){
 const [swap,setSwap]=useState(false),[limit,setLimit]=useState('200'),[lines,setLines]=useState(true),[only,setOnly]=useState(false),[selection,setSelected]=useState<number|null>(null),[sync,setSync]=useState(false),[original,setOriginal]=useState(false),[track,setTrack]=useState<string|null>(null),[message,setMessage]=useState('');
 const [trackChoices,setTrackChoices]=useState<string[]>([]);
 const [comparisonState,setComparisonState]=useState('all'),[page,setPage]=useState(0);
 const [appearance,setAppearance]=useState(()=>({...savedAppearance}));
 useEffect(()=>{savedAppearance={...appearance}},[appearance]);
 const a=swap?pair[1]:pair[0],b=swap?pair[0]:pair[1],key=generation+':'+component+':'+a+':'+b+':'+comparison+':'+scope+':'+comparisonState+':'+page;
 const {data,error}=useLoad<MatchData>(comparison?`/api/comparison/pairs/${a}/${b}?scope=${scope}&component=${component}&state=${comparisonState}&limit=200&offset=${page*200}`:`/api/matches/${a}/${b}?component=${component}`),detail=useLoad<Track>(track==null?null:`/api/tracks/${encodeURIComponent(track)}?component=${component}`);
 const selected=selection!==null&&data?.pairs[selection]?selection:null;
 const left=useImage(url(`/api/images/${a}/asset?size=${original?'original':'preview'}`)),right=useImage(url(`/api/images/${b}/asset?size=${original?'original':'preview'}`));
 const [views,setViews]=useState<View[]>([{scale:null,cx:0,cy:0},{scale:null,cx:0,cy:0}]),[size,setSize]=useState({w:1000,h:600}),[dragging,setDragging]=useState(false);
 const frame=useRef<HTMLDivElement>(null),host=useRef<HTMLDivElement>(null),canvas=useRef<HTMLCanvasElement>(null),loaded=useRef(''),request=useRef(0),drag=useRef<{side:number;x:number;y:number;view:View;s:number;moved:boolean}|null>(null);
 useEffect(()=>{request.current++;setTrack(null);setTrackChoices([]);setMessage('');loaded.current=''},[key]);
 useEffect(()=>{if(!data)return;const cache=saved.get(key);setViews(cache?.views||[{scale:null,cx:data.width_a/2,cy:data.height_a/2},{scale:null,cx:data.width_b/2,cy:data.height_b/2}]);setSelected(cache?.selected!=null&&cache.selected<data.count?cache.selected:null);loaded.current=key},[data]);
 useEffect(()=>{if(loaded.current===key){saved.set(key,{views,selected});if(saved.size>30)saved.delete(saved.keys().next().value!)}},[views,selected]);
 useEffect(()=>{const el=host.current;if(!el)return;const r=new ResizeObserver(()=>setSize({w:el.clientWidth,h:el.clientHeight}));r.observe(el);return()=>{r.disconnect();request.current++}},[]);
 const dimensions=(side:number)=>side===0?[data!.width_a,data!.height_a]:[data!.width_b,data!.height_b];
 function transform(side:number){const rawDimensions=dimensions(side),[w,h]=rawDimensions.map(value=>Math.max(1,value)),v=views[side],s=v.scale??Math.min((size.w/2-24)/w,(size.h-24)/h);return {s,x:side*size.w/2+size.w/4-v.cx*s,y:size.h/2-v.cy*s}}
 function position(side:number,i:number){const [w,h]=dimensions(side),t=transform(side),[u,v]=xy(side===0?data!.a_points[i]:data!.b_points[i],w,h);return [u*t.s+t.x,v*t.s+t.y]}
 const n=data?limit==='all'?data.count:Math.min(data.count,+limit):0;
 const indices=data?(only?(selected==null?[]:[selected]):Array.from({length:n},(_,i)=>Math.floor(i*data.count/n))):[];
 useEffect(()=>{const c=canvas.current;if(!c||!data)return;const dpr=Math.min(devicePixelRatio,2);c.width=size.w*dpr;c.height=size.h*dpr;const ctx=c.getContext('2d')!;ctx.scale(dpr,dpr);ctx.fillStyle='#0b1419';ctx.fillRect(0,0,size.w,size.h);
 [left.image,right.image].forEach((im,side)=>{const t=transform(side),[w,h]=dimensions(side);ctx.save();ctx.beginPath();ctx.rect(side*size.w/2,0,size.w/2,size.h);ctx.clip();if(im)ctx.drawImage(im,t.x,t.y,w*t.s,h*t.s);ctx.restore()});ctx.strokeStyle='#536872';ctx.beginPath();ctx.moveTo(size.w/2,0);ctx.lineTo(size.w/2,size.h);ctx.stroke();
 const draw=(i:number,active=false)=>{
  const [x1,y1]=position(0,i),[x2,y2]=position(1,i),visible=[x1>=0&&x1<size.w/2&&y1>=0&&y1<size.h,x2>=size.w/2&&x2<size.w&&y2>=0&&y2<size.h];
  const color=active?'#fff3a6':comparison?(data.available?(data.retained?.[i]?'#71e8c3':'#ffbd61'):'#88969f'):`hsl(${i*47%360},85%,72%)`;
  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=active?Math.max(2,appearance.lineWidth+1):appearance.lineWidth;ctx.globalAlpha=appearance.lineOpacity;
  if((lines||active)&&visible.every(Boolean)){ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke()}
  ctx.globalAlpha=appearance.pointOpacity;ctx.fillStyle=color;
  [[x1,y1],[x2,y2]].forEach(([x,y],side)=>{if(!visible[side])return;ctx.beginPath();ctx.arc(x,y,appearance.pointSize/2+(active?2:0),0,Math.PI*2);ctx.fill();if(active){ctx.strokeStyle='#17242b';ctx.lineWidth=1.5;ctx.stroke()}});
  ctx.restore();
 };indices.forEach(i=>{if(i!==selected)draw(i)});if(selected!==null)draw(selected,true);
 },[data,left.image,right.image,size,views,limit,lines,only,selected,appearance]);
 function zoom(side:number,factor:number,px:number,py:number){if(!data)return;setViews(vs=>vs.map((v,i)=>{if(i!==side&&!sync)return v;const t=transform(i),[w,h]=dimensions(i),fit=Math.min((size.w/2-24)/w,(size.h-24)/h),s=Math.max(fit*.25,Math.min(16,t.s*factor)),dx=i===side?px-(side*size.w/2+size.w/4):0,dy=i===side?py-size.h/2:0;return {cx:v.cx+dx*(1/t.s-1/s),cy:v.cy+dy*(1/t.s-1/s),scale:s}}))}
 useEffect(()=>{const c=canvas.current;if(!c)return;const wheel=(e:WheelEvent)=>{e.preventDefault();const r=c.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;zoom(x<size.w/2?0:1,Math.exp(-Math.max(-100,Math.min(100,e.deltaY))*.002),x,y)};c.addEventListener('wheel',wheel,{passive:false});return()=>c.removeEventListener('wheel',wheel)},[data,size,views,sync]);
 function locate(){if(!data||selected===null)return;setOriginal(true);setViews([0,1].map(side=>{const [w,h]=dimensions(side),[cx,cy]=xy(side===0?data.a_points[selected]:data.b_points[selected],w,h);return {cx,cy,scale:1}}))}
 async function inspectTrack(){
  if(!data||selected===null)return;
  setTrackChoices([]);setMessage('');
  if(data.tracks){setTrack(data.tracks[selected]);return;}
  const token=++request.current;
  const [af,bf]=data.pairs[selected];
  setMessage('查询两个端点共同关联的 Track…');
  try{
   const [left,right]=await Promise.all([
    get<FeatureData>(`/api/images/${a}/features?component=${component}`),
    get<FeatureData>(`/api/images/${b}/features?component=${component}`)
   ]);
   if(token!==request.current)return;
   const matches=commonMatchTracks(left.observations,right.observations,af,bf);
   setTrack(matches.length===1?matches[0]:null);
   setTrackChoices(matches.length>1?matches:[]);
   setMessage(matches.length===0?'这对匹配的两个端点没有共同关联的 Track':matches.length>1?'找到多个共同 Track，请选择：':'');
  }catch(e){if(token===request.current)setMessage(String(e))}
 }

 return <Dialog title="双图匹配" onClose={onClose}><div className="match-workbench" ref={frame}>
 <header className="viewer-heading"><div><strong>{comparison?'Track 共视对照':'双图匹配'} · {fmt(data?.total??data?.count)} 对</strong><span>{comparison?'绿：最终保留 / 黄：输入关系未保留；不是直接匹配文件':'匹配阶段 · 几何验证后保存的直接匹配'}</span></div><button onClick={()=>fullscreen(frame.current)}>浏览器全屏</button><button aria-label="关闭匹配" onClick={onClose}>关闭 ×</button></header>
 {comparison&&<div className="match-toolbar"><select aria-label="共视对照集合" value={comparisonState} onChange={e=>{setComparisonState(e.target.value);setPage(0)}}><option value="all">两阶段关联并集</option><option value="final">最终保留共视</option><option value="lost">未保留输入共视</option></select><button disabled={!page} onClick={()=>setPage(v=>v-1)}>上一页</button><span>第 {page+1} 页 · 每页最多 200 对</span><button disabled={(page+1)*200>=(data?.total||0)} onClick={()=>setPage(v=>v+1)}>下一页</button>{data?.available===false&&<span>最终对照不可用，不能判断删除。</span>}</div>}
 <div className="match-toolbar"><label>连线数量<select aria-label="连线数量" value={limit} onChange={e=>setLimit(e.target.value)}><option value="100">100</option><option value="200">200</option><option value="500">500</option><option value="all">全部</option></select></label><label><input type="checkbox" checked={lines} onChange={e=>setLines(e.target.checked)}/>显示连线</label><label><input type="checkbox" checked={only} onChange={e=>setOnly(e.target.checked)}/>仅选中匹配</label><label><input type="checkbox" checked={sync} onChange={e=>setSync(e.target.checked)}/>联动缩放</label><button onClick={()=>{setSwap(!swap);setViews(v=>[v[1],v[0]])}}>交换</button><button onClick={()=>{if(data)setViews([{scale:null,cx:data.width_a/2,cy:data.height_a/2},{scale:null,cx:data.width_b/2,cy:data.height_b/2}])}}>适应窗口</button><button onClick={()=>{setOriginal(true);setViews(v=>v.map(x=>({...x,scale:1})))}}>原图 100%</button><strong>显示 {fmt(new Set([...indices,...selected===null?[]:[selected]]).size)} / {fmt(data?.count)}</strong></div>
 <div className="match-appearance" aria-label="匹配显示设置">
 <label title="0% 完全透明，100% 不透明">连线不透明度<input aria-label="匹配连线不透明度" type="range" min="0" max="1" step=".05" value={appearance.lineOpacity} onChange={e=>setAppearance(v=>({...v,lineOpacity:+e.target.value}))}/><output>{Math.round(appearance.lineOpacity*100)}%</output></label>
 <label>连线宽度<input aria-label="匹配连线宽度" type="range" min=".5" max="5" step=".1" value={appearance.lineWidth} onChange={e=>setAppearance(v=>({...v,lineWidth:+e.target.value}))}/><output>{appearance.lineWidth.toFixed(1)} px</output></label>
 <label title="0% 完全透明，100% 不透明">点不透明度<input aria-label="匹配点不透明度" type="range" min="0" max="1" step=".05" value={appearance.pointOpacity} onChange={e=>setAppearance(v=>({...v,pointOpacity:+e.target.value}))}/><output>{Math.round(appearance.pointOpacity*100)}%</output></label>
 <label>点直径<input aria-label="匹配点直径" type="range" min="2" max="16" step="1" value={appearance.pointSize} onChange={e=>setAppearance(v=>({...v,pointSize:+e.target.value}))}/><output>{appearance.pointSize} px</output></label>
 <button onClick={()=>setAppearance({...defaultAppearance})}>恢复显示默认值</button>
 </div>
 <div className="pair-labels"><span title={data?.name_a}>{data?.name_a||`图像 #${a}`}</span><span title={data?.name_b}>{data?.name_b||`图像 #${b}`}</span></div>
 <div className="match-body"><div className="match-canvas-wrap" ref={host}><canvas ref={canvas} aria-label="双图匹配画布" data-views={JSON.stringify(views)} style={{cursor:dragging?'grabbing':'grab'}} onPointerDown={e=>{if(!data||e.button!==0)return;const r=e.currentTarget.getBoundingClientRect(),side=e.clientX-r.left<size.w/2?0:1;e.currentTarget.setPointerCapture(e.pointerId);drag.current={side,x:e.clientX,y:e.clientY,view:views[side],s:transform(side).s,moved:false}}} onPointerMove={e=>{const d=drag.current;if(!d||!data)return;const dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.hypot(dx,dy)>5)d.moved=true;if(d.moved){setDragging(true);const [w,h]=dimensions(d.side);setViews(vs=>vs.map((v,i)=>i===d.side?{...d.view,cx:Math.max(0,Math.min(w,d.view.cx-dx/d.s)),cy:Math.max(0,Math.min(h,d.view.cy-dy/d.s))}:v))}}} onPointerUp={e=>{const d=drag.current;drag.current=null;setDragging(false);if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);if(!d||d.moved||!data||appearance.pointOpacity===0)return;const r=e.currentTarget.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best:number|null=null,distance=Math.max(10,appearance.pointSize/2+2);new Set([...indices,...selected===null?[]:[selected]]).forEach(i=>{const [px,py]=position(d.side,i),dd=Math.hypot(px-x,py-y);if(dd<distance){distance=dd;best=i}});setSelected(best);request.current++;setTrack(null);setTrackChoices([]);setMessage('')}} onPointerCancel={()=>{drag.current=null;setDragging(false)}}/>
 {(error||left.error||right.error)&&<div className="canvas-message error">{error||left.error||right.error}</div>}{(!data||!left.image||!right.image)&&!error&&!left.error&&!right.error&&<div className="canvas-message">加载匹配与图像…</div>}</div>
 {track&&<aside className="match-track"><header><strong>Track {track}</strong><button onClick={()=>setTrack(null)}>返回匹配 ×</button></header><p>输入 {detail.data?.length} → 最终 {detail.data?.final_length} 次观测</p>{detail.error?<p className="inline-error">{detail.error}</p>:detail.loading?'加载 Track…':detail.data?.observations.map(o=><button key={`${o.image}:${o.feature}:${o.retained}`} onClick={()=>{onClose();onObservation?.(o)}}><img src={url(`/api/images/${o.image}/asset?size=thumb`)}/><span title={o.name}>{o.name.split('/').pop()}</span><small>特征 #{o.feature} · {fateName(o.fate)}</small></button>)}</aside>}</div>
 <footer className="viewer-help"><span>{selected!==null&&data?`特征 ${data.pairs[selected][0]} ↔ ${data.pairs[selected][1]}`:'滚轮缩放 · 拖动平移 · 点击端点选择匹配'}</span>{selected!==null&&<><button onClick={locate}>定位两个端点</button><button onClick={inspectTrack}>查看关联 Track</button></>}{message&&<span>{message}</span>}{trackChoices.map(id=><button key={id} onClick={()=>setTrack(id)}>Track {id}</button>)}{!!data?.invalid&&<span>跳过 {data.invalid} 条越界索引</span>}</footer>
 </div></Dialog>
}
