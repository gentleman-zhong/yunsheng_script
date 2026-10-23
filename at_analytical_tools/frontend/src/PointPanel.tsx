import {useEffect} from 'react';
import {Observation,PointDetail,fmt,useLoad} from './api';
import {ObservationCards} from './TrackDetail';
export default function PointPanel({idx,component,onObservation,onLocate,onCameras}:{idx:number;component:number;onObservation:(o:Observation)=>void;onLocate:()=>void;onCameras:(ids:number[])=>void}){
 const {data,error,loading}=useLoad<PointDetail>(`/api/scene/${component}/point/${idx}`);
 useEffect(()=>onCameras(data?.observations.map(o=>o.image)||[]),[data]);
 if(!data)return <div className="panel-pad">{error||(loading?'加载三维点…':'')}</div>;
 return <div className="point-detail"><div className="point-heading"><span className="eyebrow">FINAL RECONSTRUCTION</span><h2>三维点 {data.track}</h2><button onClick={onLocate}>定位三维点</button></div>
 <p className="hint">最终保留观测 · 原图像素 · {data.stats.count} 次可计算观测</p>
 {(data.source_status!=='ready'||data.ambiguous)&&<p className="inline-error">{data.ambiguous?'分量归属存在歧义，完整点误差不可用。':'未提供可读的最终 Tracks，最终误差不可用。'}</p>}
 <div className="mini-metrics">{[['平均',data.stats.mean],['中位',data.stats.median],['RMSE',data.stats.rmse],['最大',data.stats.max]].map(([label,value])=><div className="metric" key={label}><span>{label}误差</span><strong>{fmt(value as number|null,3)}<small>px</small></strong></div>)}</div>
 <div className="coordinate">三维坐标<br/>{[data.x,data.y,data.z].map(v=>fmt(v,3)).join(' / ')}</div>
 <ObservationCards observations={data.observations} onObservation={onObservation}/>
 <details className="reference-observations"><summary>未纳入最终统计的输入观测 · {data.reference_observations.length}</summary><p className="hint">这里只比较观测集合，参考残差使用同一套最终相机与点坐标，不代表优化前误差。</p><ObservationCards observations={data.reference_observations} onObservation={onObservation}/></details>
 </div>;
}
