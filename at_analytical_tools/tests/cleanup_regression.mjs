import {build} from '../frontend/node_modules/esbuild/lib/main.js';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';

const dir = await mkdtemp(join(tmpdir(), 'at-cleanup-'));
try {
  const require = createRequire(import.meta.url);
  async function load(name) {
    const output = join(dir, `${name}.cjs`);
    await build({entryPoints: [new URL(`../frontend/src/${name}.tsx`, import.meta.url).pathname], bundle: true, platform: 'node', format: 'cjs', outfile: output});
    return require(output);
  }
  const output = join(dir, 'matchTracks.cjs');
  await build({entryPoints: [new URL('../frontend/src/matchTracks.ts', import.meta.url).pathname], bundle: true, platform: 'node', format: 'cjs', outfile: output});
  const {commonMatchTracks} = require(output);
  const raw = (track, feature) => ({track, feature, input_track: true, retained: false, fate: 'no_final_point', idx: null});
  const final = (track, feature) => ({track, feature, input_track: false, retained: true, fate: 'added', idx: 0});
  assert.deepEqual(commonMatchTracks([raw('1', 2)], [raw('1', 8)], 2, 3), []);
  assert.deepEqual(commonMatchTracks([raw('1', 2)], [raw('2', 3)], 2, 3), []);
  assert.deepEqual(commonMatchTracks([raw('1', 2)], [final('1', 3)], 2, 3), []);
  assert.deepEqual(commonMatchTracks([raw('10', 2), raw('2', 2), raw('2', 2)], [raw('2', 3), raw('10', 3)], 2, 3), ['2', '10']);
  assert.deepEqual(commonMatchTracks([final('1', 2)], [final('1', 3)], 2, 3), ['1']);
  const {migrationIndex} = await load('MigrationMatrix');
  const indexed = migrationIndex([{raw: 1000000, final: 999999, count: 3}, {raw: 2, final: 0, count: 4}, {raw: 2, final: 0, count: 5}]);
  assert.deepEqual(indexed.columns, [2, 1000000]);
  assert.deepEqual(indexed.rows, [0, 999999]);
  assert.equal(indexed.counts.get('2:0'), 9);
  assert.equal(indexed.maximum, 9);
  assert.deepEqual(migrationIndex([]).columns, []);
  console.log('PASS exact matching endpoints, source separation, sparse migration lengths and aggregation');
} finally {
  await rm(dir, {recursive: true, force: true});
}
