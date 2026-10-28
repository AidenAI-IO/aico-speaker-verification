'use strict';
const $ = id => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`页面文件版本不一致：缺少 ${id}，请刷新页面并检查部署文件`);
  return element;
};
let pending = false;
let current = null;
let actionError = '';
let revision = 0;
let savedReferenceId = null;
const labels = {idle: '空闲', enrolling: '正在录制参考声音，请说话', processing: '正在计算参考声纹', ready: '参考声音已就绪', monitoring: '持续比较中', error: '发生错误，已停止'};

function render(data) {
  current = data;
  const source = data.capture_source === 'raw_alsa' ? 'ALSA 直采 + 50 Hz 去直流（绕过 VQE）' : 'audio_service 麦克风（VQE 处理）';
  const model = 'ERes2Net-base · 512维';
  $('source').textContent = `模型：${model}；采集来源：${source}`;
  $('enroll').textContent = '1 · 录制参考声音（4 秒）';
  $('enroll-description').textContent = '参考录制共 4 秒。模型会从完整录音取前、中、后三个 2.015 秒窗口，分别提取声纹并做归一化平均；太安静的窗口会跳过。请全程自然说话，避免旁人同时说话。' + (data.reference_saving_enabled
    ? '完整 4 秒 WAV 会保存在板子私有目录供诊断，即使注册失败也会保存。'
    : '本服务未启用诊断录音保存；推理临时文件会在处理后删除。');
  const warming = data.capture_source === 'raw_alsa' && data.state === 'enrolling' && data.enroll_remaining >= 4;
  $('state').textContent = `${warming ? '麦克风预热中，请稍等…' : labels[data.state] || data.state} · 板载麦克风 ${data.recorder_active ? '采集中' : '未开启'}`;
  $('indicator').classList.toggle('active', data.recorder_active);
  $('countdown').textContent = data.state === 'enrolling' ? `剩余 ${data.enroll_remaining.toFixed(1)} 秒` : '';
  $('score').textContent = data.score === null ? '—' : data.score.toFixed(3);
  $('detail').textContent = data.latest_timestamp === null ? '暂无当前分数。静音、停止、等待和失败窗口不会产生分数。' : `窗口时间 ${new Date(data.latest_timestamp * 1000).toLocaleTimeString()} · 仅显示相似度，不代表接受或拒绝`;
  $('rms').textContent = data.rms === null ? '—' : data.rms.toFixed(4);
  $('cutoff').textContent = data.min_rms.toFixed(4);
  $('error').textContent = actionError || data.error || '';
  const recording = data.reference_recording;
  const nextReferenceId = recording ? recording.id : null;
  if (savedReferenceId !== nextReferenceId) {
    $('reference-player').pause();
    $('reference-player').removeAttribute('src');
    $('reference-player').load();
    savedReferenceId = nextReferenceId;
  }
  $('reference-play').disabled = !savedReferenceId;
  $('reference-info').textContent = recording ? `参考音频已保存（ID：${recording.id}，${recording.duration.toFixed(1)} 秒）。完整 RMS：${recording.full_rms.toFixed(4)}；${recording.eligible_windows} 个窗口合格；峰值：${recording.peak.toFixed(3)}。注册策略：逐窗声纹归一化后等权平均，再归一化。` : '';
  const active = ['enrolling', 'processing', 'monitoring'].includes(data.state);
  $('enroll').disabled = pending || active || data.recorder_active;
  $('start').disabled = pending || active || data.recorder_active || !data.enrolled;
  $('stop').disabled = pending;
  $('reset').disabled = pending;
  let path = '';
  let gap = true;
  const points = data.history;
  points.forEach((point, index) => {
    if (point.score === null) { gap = true; return; }
    const x = points.length === 1 ? 0 : index * 600 / (points.length - 1);
    const y = (1 - point.score) * 70;
    path += `${gap ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)} `;
    gap = false;
  });
  $('trace').setAttribute('d', path);
}

$('reference-play').addEventListener('click', () => {
  if (savedReferenceId) {
    $('reference-player').src = `/api/reference/${savedReferenceId}`;
    $('reference-player').play().catch(error => { $('error').textContent = error.message; });
  }
});

async function request(path, options = {}) {
  const response = await fetch(path, {...options, cache: 'no-store', signal: AbortSignal.timeout(5000)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

for (const action of ['enroll', 'start', 'stop', 'reset']) {
  $(action).addEventListener('click', async () => {
    pending = true;
    revision += 1;
    actionError = '';
    try {
      if (current) render(current);
      render(await request(`/api/${action}`, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Speaker-Gate': '1'}, body: '{}'}));
    } catch (error) {
      actionError = error.message;
      if (current) current = {...current, score: null, latest_timestamp: null};
      $('error').textContent = actionError;
    } finally {
      pending = false;
      if (current) render(current);
    }
  });
}

async function poll() {
  try {
    if (!pending) {
      const started = revision;
      const data = await request('/api/status');
      if (started === revision && !pending) render(data);
    }
  } catch (error) {
    $('score').textContent = '—';
    $('state').textContent = '连接断开，麦克风状态未知';
    $('error').textContent = `${error.message}。45 秒没有收到网页心跳后，服务会自动停止采集。`;
    $('enroll').disabled = true;
    $('start').disabled = true;
  } finally {
    setTimeout(poll, 500);
  }
}
poll();
