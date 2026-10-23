import {chromium} from '../frontend/node_modules/@playwright/test/index.mjs';
import fs from 'node:fs/promises';
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
try{
 const page=await browser.newPage({viewport:{width:1920,height:1080}});
 await page.goto(process.env.AT_VIEWER_TEST_URL||'http://127.0.0.1:8081');
 await page.locator('.scene-canvas canvas').waitFor();await page.locator('.scene-notice').waitFor({state:'hidden'});
 async function measure(){return page.evaluate(()=>new Promise(resolve=>{const deltas=[];let start=0,last=0;function tick(t){if(!start){start=t;last=t;requestAnimationFrame(tick);return}deltas.push(t-last);last=t;if(t-start<3000){requestAnimationFrame(tick);return}const sorted=[...deltas].sort((a,b)=>a-b);resolve({frames:deltas.length,duration_ms:t-start,raf_fps:1000*deltas.length/(t-start),p95_frame_ms:sorted[Math.floor(sorted.length*.95)]})}requestAnimationFrame(tick)}))}
 const allFrustums=await measure();
 const result={renderer:'headless Chromium SwiftShader (software)',viewport:'1920x1080',method:'3 seconds requestAnimationFrame intervals while the visible 3D scene renders continuously; excludes input latency and does not represent user GPU performance',allFrustums};
 await fs.writeFile(new URL('../validation/v2/render-performance.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser.close()}
