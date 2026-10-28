// UI contract and button-flow regression. No network, microphone or npm dependencies.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..', 'speaker_verification', 'web_demo_static');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(ids).size, ids.length, 'HTML IDs must be unique');
for (const [, id] of script.matchAll(/\$\('([^']+)'\)/g)) {
  assert.ok(ids.includes(id), `app.js references missing HTML element ${id}`);
}

function element() {
  return {textContent: '', disabled: false, listeners: {}, classList: {toggle() {}},
    pause() {}, load() {}, removeAttribute() {}, setAttribute() {},
    play() { return Promise.resolve(); },
    addEventListener(name, fn) { this.listeners[name] = fn; }};
}
const state = {
  state: 'idle', enrolled: false, recorder_active: false, score: null,
  latest_timestamp: null, reference_recording: null, history: [], error: null,
  rms: null, min_rms: .005, capture_source: 'raw_alsa', speaker_model: 'eres2net',
  enrollment_mode: 'fixed', enroll_remaining: 0, reference_saving_enabled: false,
};
function setup(removeId, initialState = state) {
  const elements = new Map(ids.filter(id => id !== removeId).map(id => [id, element()]));
  const requests = [];
  const context = {
    document: {getElementById(id) { return elements.get(id) || null; }},
    AbortSignal, setTimeout() {}, Date, console,
    async fetch(url, options) {
      requests.push({url, options});
      return {ok: true, async json() {
        return url === '/api/enroll'
          ? {...initialState, state: 'enrolling', recorder_active: true, enroll_remaining: 4}
          : {...initialState};
      }};
    },
  };
  vm.runInNewContext(script, context);
  return {elements, requests};
}
async function main() {
  const app = setup();
  await new Promise(setImmediate); // allow initial status request + render
  assert.match(app.elements.get('state').textContent, /空闲/);
  assert.match(app.elements.get('source').textContent, /ERes2Net-base · 512维/);
  assert.match(app.elements.get('enroll-description').textContent, /未启用/);
  assert.equal(app.elements.get('reference-play').disabled, true);
  assert.match(app.elements.get('enroll-description').textContent, /4 秒/);
  assert.equal(app.elements.get('enroll').disabled, false);
  assert.equal(app.elements.get('start').disabled, true);
  await app.elements.get('enroll').listeners.click();
  assert.equal(app.requests.at(-1).url, '/api/enroll');
  assert.equal(app.requests.at(-1).options.method, 'POST');
  assert.equal(app.requests.at(-1).options.headers['X-Speaker-Gate'], '1');
  assert.equal(app.elements.get('enroll').disabled, true);
  assert.match(app.elements.get('state').textContent, /预热/);
  assert.equal(app.elements.get('stop').disabled, false, 'pending must clear after enrollment POST');

  const recording = {id: 'saved-reference', duration: 4, full_rms: .1, eligible_windows: 3, peak: .5};
  const saved = setup(undefined, {...state, state: 'ready', enrolled: true,
    reference_saving_enabled: true, reference_recording: recording});
  await new Promise(setImmediate);
  assert.equal(saved.elements.get('reference-play').disabled, false);
  assert.match(saved.elements.get('enroll-description').textContent, /私有目录/);
  saved.elements.get('reference-play').listeners.click();
  assert.equal(saved.elements.get('reference-player').src, '/api/reference/saved-reference');

  // Exact deployed failure: new JS with HTML lacking enroll-description.
  const mismatched = setup('enroll-description');
  await new Promise(setImmediate);
  assert.match(mismatched.elements.get('error').textContent, /页面文件版本不一致/);
  assert.equal(mismatched.elements.get('enroll').disabled, true);
  console.log('PASS: DOM contract, ERes2Net idle rendering, enrollment and playback clicks, mismatched asset error');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
