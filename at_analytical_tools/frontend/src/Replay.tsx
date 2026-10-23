import {useEffect,useMemo,useState} from 'react';
import {get,url,useLoad} from './api';
export interface ReplayImage{name:string;id:number|null;visible:boolean}
export interface ReplayEvent{kind:'initial'|'add';names:string[];images:ReplayImage[];line:number|null;timestamp:string|null;excerpt:string|null;successful:boolean;ordinal_start:number|null;ordinal_end:number|null;cumulative:number}
export interface ReplayBranch{key:string;source_id:string;run_id:string;branch_id:string;path:string;source_kind:string;fingerprint:string;playable:boolean;initial_confirmed:boolean;sequence_complete:boolean;cross_checked:boolean;warnings:string[];coverage:number;registered_images:number;image_count:number;event_count:number;time:string|null;initial_pair?:string[];events?:ReplayEvent[];failed:boolean}
interface ReplayCatalog{recommended:string|null;entries:ReplayBranch[];sources:{source_id:string;path:string;kind:string;status:string;error?:string}[]}
export function replayFrame(events:ReplayEvent[],step:number){
 const prefix=events.slice(0,Math.max(0,Math.min(step,events.length)));
 const ids=(e:ReplayEvent)=>e.images.filter(i=>i.visible&&i.id!==null).map(i=>i.id!);
 return {visible:new Set(prefix.flatMap(ids)),seed:new Set(prefix.filter(e=>e.kind==='initial').flatMap(ids)),current:new Set(prefix.length?ids(prefix[prefix.length-1]):[])};
}
export function eventLabel(e:ReplayEvent){return e.kind==='initial'?(e.successful?'初始对 · '+e.names.length+' 张':'初始对尝试 · 未确认成功'):e.ordinal_start==null?'加入记录 · '+e.names.length+' 张':e.ordinal_start===e.ordinal_end?`第 ${e.ordinal_start} 张`:`第 ${e.ordinal_start}–${e.ordinal_end} 张 · 同时加入`}
export function useReplay(component:number,active:boolean){
 const [open,setOpen]=useState(false),[scan,setScan]=useState(0),[choice,setChoice]=useState(''),[playing,setPlaying]=useState(false),[step,setStep]=useState(0),[speed,setSpeed]=useState(1),[background,setBackground]=useState(false),[stale,setStale]=useState(''),[inspect,setInspect]=useState<number|null>(null),[focusToken,setFocusToken]=useState(0);
 const catalog=useLoad<ReplayCatalog>(open?`/api/replay/sources?component=${component}&refresh=${scan}`:null);
 useEffect(()=>{setOpen(false);setChoice('');setPlaying(false);setStep(0);setBackground(false);setStale('');setInspect(null)},[component]);
 useEffect(()=>{if(catalog.data){setChoice(catalog.data.recommended||catalog.data.entries[0]?.key||'');setPlaying(false);setStep(0)}},[catalog.data]);
 const source=catalog.data?.entries.find(b=>b.key===choice);
 const path=source?`/api/replay/events?component=${component}&source_id=${source.source_id}&run_id=${source.run_id}&branch_id=${source.branch_id}&fingerprint=${source.fingerprint}`:null;
 const detail=useLoad<ReplayBranch>(open?path:null);
 const events=useMemo(()=>detail.data?.events?.filter(e=>e.successful)||[],[detail.data]);
 const running=!!(open&&detail.data?.playable&&!stale&&!detail.error);
 const frame=useMemo(()=>replayFrame(events,step),[events,step]);
 useEffect(()=>{setStep(0);setPlaying(false);setInspect(null);setStale('')},[choice]);
 useEffect(()=>{if(!active)setPlaying(false)},[active]);
 useEffect(()=>{const pause=()=>{if(document.hidden)setPlaying(false)};document.addEventListener('visibilitychange',pause);return()=>document.removeEventListener('visibilitychange',pause)},[]);
 useEffect(()=>{if(!playing||!active||!running)return;if(step>=events.length){setPlaying(false);return}const timer=setTimeout(()=>{if(document.hidden||document.querySelector('[aria-modal="true"]')){setPlaying(false);return}setStep(n=>Math.min(n+1,events.length))},1000/speed);return()=>clearTimeout(timer)},[playing,active,running,step,speed,events.length]);
 // Revalidate evidence without resetting the timeline. A changed file must be reloaded explicitly.
 useEffect(()=>{
  if(!open||!path||!detail.data)return;
  const controller=new AbortController();
  const expected=JSON.stringify(detail.data);
  let timer:ReturnType<typeof setTimeout>;
  async function validate(){
   try{
    if(document.hidden)return;
    const fresh=await get<ReplayBranch>(path!,controller.signal);
    if(!controller.signal.aborted&&JSON.stringify(fresh)!==expected){setPlaying(false);setStale('过程记录的证据或映射已更新，请重新扫描。')}
   }catch(error){
    if(!controller.signal.aborted){setPlaying(false);setStale(String(error))}
   }finally{
    if(!controller.signal.aborted)timer=setTimeout(validate,10000);
   }
  }
  timer=setTimeout(validate,10000);
  return()=>{clearTimeout(timer);controller.abort()};
 },[open,path,detail.data]);

 function seek(n:number){setPlaying(false);setInspect(null);setStep(Math.max(0,Math.min(n,events.length)))}
 function close(){setOpen(false);setPlaying(false);setStep(0);setBackground(false);setInspect(null);setStale('')}
 return {open,setOpen,catalog,source,detail,events,choice,choose:(value:string)=>{setPlaying(false);setChoice(value)},playing,play:()=>{if(!running)return;if(step>=events.length)setStep(0);setPlaying(v=>!v)},pause:()=>setPlaying(false),step,seek,speed,setSpeed,background,setBackground,stale,running,frame,inspect,inspectImage:(id:number)=>{setPlaying(false);setInspect(id)},focusToken,locate:()=>{setPlaying(false);setFocusToken(n=>n+1)},close,rescan:()=>{setPlaying(false);setStep(0);setChoice('');setStale('');setScan(n=>n+1)}};
}
export type ReplayController=ReturnType<typeof useReplay>;
const sourceStatus:Record<string,string>={non_incremental:'非增量式运行，不包含增量加入过程',unsupported:'未识别到支持的增量过程格式',unreadable:'来源不可读',ready:'已识别过程记录'};
export function ReplayPanel({replay:r}:{replay:ReplayController}){
 const e=r.events[r.step-1],data=r.detail.data;
 const inspection=r.inspect==null?[]:r.events.flatMap(e=>e.images).filter(i=>i.id===r.inspect).slice(0,1);
 const images=inspection.length?inspection:e?.images||[];
 return <section className={'replay-panel'+(r.open?' is-open':'')} aria-label="增量式重建回放"><div className="replay-heading"><button onClick={()=>r.open?r.close():r.setOpen(true)}>{r.open?'退出回放':'复现增量式重建过程'}</button>{r.open&&<><strong>加入顺序回放 · 使用最终位姿</strong><button onClick={r.rescan}>重新扫描</button></>}</div>
 {r.open&&<div className="replay-content"><p className="hint">按记录展示相机加入顺序，不代表当时的相机位置或点云。最终点云仅作背景。</p>
 {r.catalog.loading&&<p>扫描过程记录…</p>}{r.catalog.error&&<p className="inline-error">{r.catalog.error}</p>}
 {r.catalog.data&&<><label className="replay-source">回放来源<select aria-label="回放来源" value={r.choice} onChange={e=>r.choose(e.target.value)}>{r.catalog.data.entries.map(b=><option value={b.key} key={b.key}>{b.key===r.catalog.data?.recommended?'推荐 · ':''}{b.source_kind==='report'?'报告':'日志'} · {b.path} · 运行 {b.run_id}/分支 {b.branch_id} · {b.coverage}/{b.registered_images} 张 · {b.sequence_complete?'完整':'片段或未确认'} · {b.time||'时间未记录'}</option>)}</select></label>
 {!r.catalog.data.entries.length&&<p className="empty">{r.catalog.data.sources.length?'未找到可回放的增量步骤，请查看来源状态。':'未找到日志或重建报告。补充记录后可重新扫描。'}</p>}
 <details className="replay-sources"><summary>来源状态 · {r.catalog.data.sources.length} 个文件</summary>{r.catalog.data.sources.map(s=><p key={s.source_id}>{s.path}：{sourceStatus[s.status]||s.status}{s.error?' · '+s.error:''}</p>)}</details></>}
 {r.detail.loading&&<p>加载加入顺序…</p>}{(r.detail.error||r.stale)&&<p className="inline-error">{r.detail.error||r.stale} 请重新扫描。</p>}
 {data&&<><div className="replay-meta"><span>初始对：{data.initial_confirmed?data.events?.find(e=>e.kind==='initial')?.names.join(' + '):'未确认'}</span><span>覆盖 {data.coverage}/{data.registered_images} 张 · {data.sequence_complete?'完整顺序':'不完整记录'}{data.cross_checked?' · 日志与报告顺序相互核对一致':''}</span></div>
 {data.warnings.length>0&&<details className="replay-warnings" open><summary>来源提示 · {data.warnings.length}</summary>{data.warnings.map((w,i)=><p key={i}>{w}</p>)}</details>}
 {!data.playable&&<p className="inline-error">当前分支无法可靠映射为三维回放；仍可查看下方事件记录。</p>}
 <div className="replay-actions"><button disabled={!r.running} onClick={r.play}>{r.playing?'暂停':'播放'}</button><button disabled={!r.running||r.step===0} onClick={()=>r.seek(r.step-1)}>上一步</button><button disabled={!r.running||r.step>=r.events.length} onClick={()=>r.seek(r.step+1)}>下一步</button><button disabled={!r.running} onClick={()=>r.seek(0)}>从头开始</button><select aria-label="回放速度" value={r.speed} onChange={e=>r.setSpeed(+e.target.value)}>{[.5,1,2,4].map(s=><option key={s} value={s}>{s}×</option>)}</select><button disabled={!r.running||!r.frame.current.size} onClick={r.locate}>定位当前步骤</button><label><input type="checkbox" checked={r.background} onChange={e=>r.setBackground(e.target.checked)}/>最终点云背景</label></div>
 <div className="replay-progress"><input aria-label="回放进度" type="range" min="0" max={r.events.length} value={r.step} disabled={!r.running} onChange={e=>r.seek(+e.target.value)}/><span>事件 {r.step}/{r.events.length} · 已加入 {e?.cumulative||0}/{data.image_count} 张 · 三维显示 {r.frame.visible.size} 张</span></div>
 <div className="replay-current"><strong>{inspection.length?'查看相机':e?eventLabel(e):'准备开始'}{r.step===r.events.length&&r.events.length>0?' · 回放结束':''}</strong><span>{e?.timestamp||'时间未记录'}{e?.line?` · 日志第 ${e.line} 行`:''}</span><div className="replay-images">{images.map((im,i)=><div key={im.name+':'+i}>{im.id!=null&&<img src={url(`/api/images/${im.id}/asset?size=thumb`)}/>}<span>{im.name}</span>{!im.visible&&<small>当前分量无可显示相机</small>}</div>)}</div></div>
 <details><summary>完整事件记录 · 点击跳转</summary><ol className="replay-events">{data.events?.map((event,i)=><li key={i}><button disabled={!r.running||!event.successful} onClick={()=>r.seek(r.events.indexOf(event)+1)}>{eventLabel(event)} · {event.names.join(' + ')}</button><small>{event.timestamp||'时间未记录'}{event.line?` · 行 ${event.line}`:''}</small><code>{event.excerpt}</code></li>)}</ol></details>
 </>}
 </div>}
 </section>;
}
