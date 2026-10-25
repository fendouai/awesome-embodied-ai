const fs=require('fs'),path=require('path');
let pw;try{pw=require('playwright')}catch{pw=require('/Users/f/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')}
(async()=>{
const browser=await pw.chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const page=await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:8767');await page.waitForFunction(()=>document.querySelector('#boutCount').textContent==='26');
await page.waitForFunction(()=>document.querySelector('#battleVideo').readyState>=2);
await page.evaluate(()=>{document.querySelector('#battleVideo').currentTime=3.8});await page.waitForFunction(()=>document.querySelector('#simTime').textContent.includes('3.80'));
const before=await page.locator('#actor0').innerText();await page.screenshot({path:'evidence/dashboard.png',fullPage:true});
await page.click('#inspectMode');await page.waitForFunction(()=>document.querySelector('#sceneImage').src.startsWith('blob:'));let src=await page.locator('#sceneImage').getAttribute('src');
await page.click('[data-camera="top"]');await page.waitForFunction(old=>document.querySelector('#sceneImage').src!==old,src);src=await page.locator('#sceneImage').getAttribute('src');
await page.locator('#timeSlider').evaluate(el=>{el.value=0;el.dispatchEvent(new Event('input',{bubbles:true}))});await page.waitForFunction(old=>document.querySelector('#sceneImage').src!==old,src);await page.waitForFunction(()=>document.querySelector('#simTime').textContent.includes('0.00'));const after=await page.locator('#actor0').innerText();if(before===after)throw Error('Telemetry did not change');
await page.screenshot({path:'evidence/replay-3d.png',fullPage:true});
await page.click('[data-tab="rankings"]');if(await page.locator('.rank-table tbody tr').count()!==5)throw Error('Dual rankings missing');await page.screenshot({path:'evidence/rankings.png',fullPage:true});
await page.click('[data-tab="protocol"]');await page.waitForFunction(()=>document.querySelector('#protocolData').textContent.includes('source_commits'));
// Exercise both manual input modes through the same UI, preserve their full sessions.
const live=[];
for(const mode of ['human-ai','human-human']){
 await page.click('[data-tab="interactive"]');await page.selectOption('#liveMode',mode);await page.click('#startLive');await page.waitForFunction(()=>document.querySelector('#liveStatus').textContent.includes('s ·'));
 await page.keyboard.down('w');if(mode==='human-human')await page.keyboard.down('ArrowUp');
 await page.waitForFunction(()=>parseFloat(document.querySelector('#liveStatus').textContent)>=1.0,{},{timeout:60000});
 await page.keyboard.up('w');await page.keyboard.up('ArrowUp');await page.click('#stopLive');await page.waitForFunction(()=>document.querySelector('#liveMetrics').textContent.includes('完整状态已保存'));
 const archive=await page.locator('#liveMetrics').innerText();if(live.includes(archive))throw Error('Reused old live session');live.push(archive);await page.screenshot({path:`evidence/${mode}.png`,fullPage:true});
}
await page.goto('http://127.0.0.1:8767/reports/report.html');await page.waitForLoadState('networkidle');
const layout=await page.locator('.page').evaluateAll(pages=>pages.map((p,i)=>{const foot=p.querySelector('footer').getBoundingClientRect();const body=[...p.children].filter(e=>!['HEADER','FOOTER'].includes(e.tagName));return {page:i+1,contentBottom:Math.max(...body.map(e=>e.getBoundingClientRect().bottom))-p.getBoundingClientRect().top,footerTop:foot.top-p.getBoundingClientRect().top,overflow:p.scrollHeight>p.clientHeight}}));
fs.writeFileSync('evidence/browser-check.json',JSON.stringify({errors,layout,live},null,2));
for(const p of layout)if(p.contentBottom>p.footerTop-8||p.overflow)throw Error('Report overflow '+JSON.stringify(p));
await page.pdf({path:'reports/report.pdf',width:'1120px',height:'1584px',printBackground:true,margin:{top:0,bottom:0,left:0,right:0},preferCSSPageSize:false});
await page.locator('.page').first().screenshot({path:'evidence/report-cover.png'});await page.locator('.page').nth(5).screenshot({path:'evidence/report-physics.png'});
if(errors.length)throw Error('Browser errors '+errors.join(';'));console.log('PASS: video, telemetry, 3D, dual rankings, protocol, both human modes, report layout/PDF');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
