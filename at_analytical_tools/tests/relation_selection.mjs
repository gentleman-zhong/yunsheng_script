import {build} from '../frontend/node_modules/esbuild/lib/main.js';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const dir=await mkdtemp(join(tmpdir(),'at-relations-'));
try{
 const output=join(dir,'relations.cjs');
 await build({entryPoints:[new URL('../frontend/src/CameraRelations.tsx',import.meta.url).pathname],bundle:true,platform:'node',format:'cjs',outfile:output});
 const {visibleRelations}=createRequire(import.meta.url)(output);
 const data={final_available:true,selected_registered:true,items:[
  {id:1,in_component:true,direct_count:50,final_count:2,kind:'both'},
  {id:2,in_component:true,direct_count:20,final_count:0,kind:'direct'},
  {id:3,in_component:true,direct_count:0,final_count:30,kind:'final'},
  {id:4,in_component:false,direct_count:100,final_count:null,kind:'unknown'},
 ]};
 assert.deepEqual(visibleRelations(data).map(r=>r.id),[1,2]);
 assert.deepEqual(visibleRelations(null),[]);
 const withoutFinal={items:data.items.map(({final_count,kind,...r})=>r),selected_registered:true};
 assert.deepEqual(visibleRelations(withoutFinal).map(r=>r.id),[1,2]);
 console.log('PASS direct matching selection, outside scope and independence from final Tracks');
}finally{await rm(dir,{recursive:true,force:true})}
