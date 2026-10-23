import {build} from '../frontend/node_modules/esbuild/lib/main.js';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const dir=await mkdtemp(join(tmpdir(),'at-replay-'));
try{
 const output=join(dir,'replay.cjs');
 await build({entryPoints:[new URL('../frontend/src/Replay.tsx',import.meta.url).pathname],bundle:true,platform:'node',format:'cjs',outfile:output});
 const {replayFrame,eventLabel}=createRequire(import.meta.url)(output);
 const im=(id,visible=true)=>({id,visible,name:`image-${id}`});
 const events=[{kind:'initial',successful:true,names:['a','b'],images:[im(1),im(2)]},{kind:'add',ordinal_start:3,ordinal_end:4,names:['c','d'],images:[im(3),im(4)]},{kind:'add',ordinal_start:5,ordinal_end:5,names:['e'],images:[im(null,false)]}];
 assert.equal(replayFrame(events,0).visible.size,0);
 assert.deepEqual([...replayFrame(events,1).visible],[1,2]);
 assert.deepEqual([...replayFrame(events,2).current],[3,4]);
 assert.deepEqual([...replayFrame(events,2).seed],[1,2]);
 assert.deepEqual([...replayFrame(events,3).visible],[1,2,3,4]);
 assert.equal(replayFrame(events,3).current.size,0);
 assert.deepEqual([...replayFrame(events,1).visible],[1,2],'Backwards seeking removes future cameras');
 assert.match(eventLabel(events[1]),/同时加入/);
 assert.match(eventLabel({...events[1],ordinal_start:null}),/加入记录/);
 assert.match(eventLabel({...events[0],successful:false}),/未确认/);
 console.log('PASS seed, grouped additions, missing geometry, backwards seek and ordinal labels');
}finally{await rm(dir,{recursive:true,force:true})}
