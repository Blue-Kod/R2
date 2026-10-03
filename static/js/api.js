// Thin fetch wrapper for the R2 Flask API. Handles JSON, errors and 401.

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request(method, url, { body, raw } = {}) {
  const opts = { method, cache: 'no-store', credentials: 'same-origin' };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch (err) {
    throw new ApiError('Нет связи с роботом', 0, null);
  }
  if (res.status === 401) {
    window.dispatchEvent(new CustomEvent('r2:unauthorized'));
    throw new ApiError('Unauthorized', 401, null);
  }
  if (raw) {
    if (!res.ok) throw new ApiError(res.statusText, res.status, null);
    return res;
  }
  const ct = res.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const message = (data && data.error) || res.statusText || 'Ошибка запроса';
    throw new ApiError(message, res.status, data);
  }
  return data;
}

export const api = {
  // --- domain helpers ---
  bootstrap: () => request('GET', '/api/bootstrap'),
  logout: () => request('POST', '/api/logout', { body: {} }),

  stats: () => request('GET', '/api/data'),
  ip: () => request('GET', '/api/ip'),

  servoAngles: () => request('GET', '/api/servo/angles'),
  setServo: (ch, angle) => request('POST', `/api/servo/${ch}/${angle}`, { body: {} }),

  calibration: () => request('GET', '/api/calibration'),
  setCalibration: (offsets, inverted) => request('POST', '/api/calibration', { body: { offsets, inverted } }),
  resetCalibration: () => request('POST', '/api/calibration/reset', { body: {} }),
  reinitServo: () => request('POST', '/api/servo/reinit', { body: {} }),
  servoPower: (enabled) => request('POST', '/api/servo/power', { body: { enabled } }),

  ikMove: (x, y, z, left) => request('POST', '/api/ik/move', { body: { x, y, z, left } }),

  pythonExec: (code) => request('POST', '/api/python/exec', { body: { code } }),

  filesList: (path) => request('GET', `/api/files?path=${encodeURIComponent(path || '')}`),
  fileRead: (path) => request('GET', `/api/files/read?path=${encodeURIComponent(path)}`),
  fileWrite: (path, content) => request('POST', '/api/files/write', { body: { path, content } }),
  fileCreate: (path, name, type) => request('POST', '/api/files/create', { body: { path, name, type } }),
  fileDelete: (path) => request('POST', '/api/files/delete', { body: { path } }),
  fileRename: (oldPath, newName) => request('POST', '/api/files/rename', { body: { old_path: oldPath, new_name: newName } }),
  fileUpload: (path, files) => {
    const fd = new FormData();
    fd.append('path', path);
    for (const f of files) fd.append('files', f);
    return fetch('/api/files/upload', { method: 'POST', body: fd, credentials: 'same-origin' }).then(async (r) => {
      const d = await r.json().catch(() => ({}));
      if (!r.ok || d.success === false) throw new ApiError(d.error || 'Ошибка загрузки', r.status, d);
      return d;
    });
  },
  fileUrl: (path) => `/api/files/download?path=${encodeURIComponent(path)}`,

  dcStatus: () => request('GET', '/api/datacollect/status'),
  dcStart: (task) => request('POST', '/api/datacollect/start', { body: { task: task || '' } }),
  dcStop: () => request('POST', '/api/datacollect/stop', { body: {} }),
  dcDatasets: () => request('GET', '/api/datacollect/datasets'),
  dcView: (episode) => request('GET', `/api/datacollect/view?episode=${episode}`),
  dcDownloadUrl: (episode) => `/api/datacollect/download?episode=${episode}`,

  cameraParams: () => request('GET', '/api/camera/params'),
  setCamera: (showLeft) => request('POST', '/api/camera/params', { body: { show_left: showLeft } }),

  update: () => request('POST', '/api/update', { body: {} }),
  reboot: () => request('POST', '/api/reboot', { body: {} }),
  shutdown: () => request('POST', '/api/shutdown', { body: {} }),
};
