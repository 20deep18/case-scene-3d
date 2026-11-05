/**
 * 协同客户端：WebSocket 自动重连 + 事件分发 + 房间状态
 */
export class CollabClient {
  constructor({ projectId, name, color, role = 'viewer', onState, onEvent, onRoster, onSystem }) {
    this.projectId = projectId;
    this.name = name;
    this.color = color;
    this.role = role;
    this.onState = onState ?? (() => {});
    this.onEvent = onEvent ?? (() => {});
    this.onRoster = onRoster ?? (() => {});
    this.onSystem = onSystem ?? (() => {});
    this.ws = null;
    this.state = 'idle';
    this.retry = 0;
    this.roster = [];
    this.you = null;
    this._pingTimer = null;
    this._lastPong = 0;
    this.latencyMs = null;
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    // 静态演示版没有 WebSocket 服务端：直接置为"离线"，不要无限重连（会把控制台刷满）
    if (typeof window !== 'undefined' && window.__DSH_STATIC__ === true) {
      this._setState('offline', '只读演示版：协同不可用');
      return;
    }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const params = new URLSearchParams({
      project: this.projectId,
      name: this.name,
      color: this.color,
      role: this.role,
    });
    const url = `${proto}://${location.host}/ws?${params}`;
    this._setState('connecting');
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      this._setState('error', err.message);
      return this._scheduleReconnect();
    }
    this.ws = ws;

    ws.onopen = () => {
      this.retry = 0;
      this._setState('online');
      this._lastPong = performance.now();
      clearInterval(this._pingTimer);
      this._pingTimer = setInterval(() => {
        if (!this.connected) return;
        this.send({ type: 'ping', t: Date.now() });
        if (this._lastPong && performance.now() - this._lastPong > 25000) {
          this.ws.close();
        }
      }, 10000);
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return;
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case 'welcome':
          this.you = msg.you;
          this.roster = msg.roster ?? [];
          this.onEvent(msg);
          this.onRoster(this.roster);
          break;
        case 'pong':
          this._lastPong = performance.now();
          this.latencyMs = Date.now() - (msg.t ?? Date.now());
          break;
        case 'roster':
          this.roster = msg.roster ?? [];
          this.onRoster(this.roster);
          break;
        case 'system':
          this.onSystem(msg);
          this.onEvent(msg);
          break;
        default:
          this.onEvent(msg);
      }
    };

    ws.onclose = () => {
      clearInterval(this._pingTimer);
      this._setState('offline');
      this._scheduleReconnect();
    };
    ws.onerror = () => {
      /* onclose 会随后触发 */
    };
  }

  _setState(state, detail = '') {
    this.state = state;
    this.onState(state, detail);
  }

  _scheduleReconnect() {
    this.retry = Math.min(this.retry + 1, 12);
    const delay = Math.min(500 * 2 ** (this.retry - 1), 15000);
    this._setState('reconnecting', `${(delay / 1000).toFixed(1)}s 后重连`);
    clearTimeout(this._reconnectTimer);
    this._reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  send(obj) {
    if (!this.connected) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }

  disconnect() {
    clearInterval(this._pingTimer);
    clearTimeout(this._reconnectTimer);
    this.retry = 99;
    this.ws?.close();
  }
}
