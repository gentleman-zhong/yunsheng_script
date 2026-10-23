import {build} from '../frontend/node_modules/esbuild/lib/main.js';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const dir=await mkdtemp(join(tmpdir(),'at-feature-types-'));
try{
 const output=join(dir,'types.cjs');await build({entryPoints:[new URL('../frontend/src/featureTypes.ts',import.meta.url).pathname],bundle:true,platform:'node',format:'cjs',outfile:output});
 const {matchesFeature,featureKind,groupFeatures}=createRequire(import.meta.url)(output);
 const rows=[{feature:0,input_track:true,idx:0,retained:true,fate:'retained',error:10},
 {feature:1,input_track:true,idx:1,retained:false,fate:'removed_observation'},
 {feature:2,input_track:true,idx:null,retained:false,fate:'no_final_point'},
 {feature:3,input_track:false,idx:3,retained:true,fate:'added',error:0}];
 const groups=groupFeatures([...rows,rows[0]]), ids=filter=>[0,1,2,3,4].filter(i=>matchesFeature(groups.get(i)||[],filter));
 assert.deepEqual(ids('all'),[0,1,2,3,4]);assert.deepEqual(ids('input'),[0,1,2]);
 assert.deepEqual(ids('reconstructed'),[0,1,3]);assert.deepEqual(ids('retained'),[0,3]);
 assert.equal(featureKind([rows[0]],3),'high');assert.equal(featureKind([rows[1]],3),'removed');
 assert.equal(featureKind([rows[2]],3),'absent');assert.equal(featureKind([],3),'none');
 const ambiguous={...rows[0],fate:'ambiguous'};assert(!matchesFeature([ambiguous],'retained'));assert.equal(featureKind([ambiguous],3),'unknown');
 assert(matchesFeature([{...rows[1],idx:0}],'reconstructed'),'Point index zero is a valid reconstructed point');
 console.log('PASS four feature filters, high errors retained, added provenance and ambiguous state');
}finally{await rm(dir,{recursive:true,force:true})}
