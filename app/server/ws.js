/**
 * 零依赖 WebSocket 服务端（RFC 6455 子集）
 * 支持：文本/二进制帧、分片、ping/pong、关闭握手。
 * 仅使用 Node 内置模块，无需 npm install。
 */
import { createHash, randomBytes } from 'node:crypto';
import { EventEmitter } from 'node:events';

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

export const OPCODE = {
  CONTINUATION: 0x0,
  TEXT: 0x1,
  BINARY: 0x2,
  CLOSE: 0x8,
  PING: 0x9,
  PONG: 0xa,
};

export class WebSocketConnection extends EventEmitter {
  constructor(socket) {
    super();
    this.socket = socket;
    this.readyState = 'open';
    this.id = randomBytes(8).toString('hex');
    this.data = {}; // 应用层挂载的用户数据（房间、昵称等）
    this._buffer = Buffer.alloc(0);
    this._fragments = [];
    this._fragmentOpcode = null;
    this.bytesIn = 0;
    this.bytesOut = 0;

    socket.on('data', (chunk) => {
      this.bytesIn += chunk.length;
      this._buffer = Buffer.concat([this._buffer, chunk]);
      try {
        this._drain();
      } catch (err) {
        this._emitError(err);
        this.close(1002, 'protocol error');
      }
    });
    socket.on('close', () => this._finish());
    socket.on('error', (err) => {
      this._emitError(err);
      this._finish();
    });
    socket.setNoDelay(true);
  }

  _emitError(err) {
    // 'error' 事件若无监听器会抛出并终止进程，这里兜底
    if (this.listenerCount('error') > 0) {
      try {
        this.emit('error', err);
      } catch {
        /* 监听器自身异常不再向上抛 */
      }
    }
  }

  _finish() {
    if (this.readyState === 'closed') return;
    this.readyState = 'closed';
    this.emit('close');
  }

  _drain() {
    while (true) {
      if (this.readyState !== 'open') return;
      if (this._buffer.length < 2) return;
      const b0 = this._buffer[0];
      const b1 = this._buffer[1];
      const fin = (b0 & 0x80) !== 0;
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let payloadLen = b1 & 0x7f;
      let offset = 2;

      if (payloadLen === 126) {
        if (this._buffer.length < offset + 2) return;
        payloadLen = this._buffer.readUInt16BE(offset);
        offset += 2;
      } else if (payloadLen === 127) {
        if (this._buffer.length < offset + 8) return;
        const big = this._buffer.readBigUInt64BE(offset);
        if (big > 64n * 1024n * 1024n) throw new Error('frame too large');
        payloadLen = Number(big);
        offset += 8;
      }
      if (payloadLen > 32 * 1024 * 1024) throw new Error('frame too large');

      let maskKey = null;
      if (masked) {
        if (this._buffer.length < offset + 4) return;
        maskKey = this._buffer.subarray(offset, offset + 4);
        offset += 4;
      }
      if (this._buffer.length < offset + payloadLen) return;

      const raw = this._buffer.subarray(offset, offset + payloadLen);
      this._buffer = this._buffer.subarray(offset + payloadLen);

      let payload = raw;
      if (masked) {
        payload = Buffer.allocUnsafe(payloadLen);
        for (let i = 0; i < payloadLen; i++) payload[i] = raw[i] ^ maskKey[i & 3];
      }

      if (opcode === OPCODE.CLOSE) {
        this.close(1000, 'bye');
        return;
      }
      if (opcode === OPCODE.PING) {
        this._sendFrame(OPCODE.PONG, payload);
        continue;
      }
      if (opcode === OPCODE.PONG) {
        continue;
      }
      if (opcode === OPCODE.TEXT || opcode === OPCODE.BINARY) {
        if (fin) {
          this._emitMessage(opcode, payload);
        } else {
          this._fragmentOpcode = opcode;
          this._fragments = [payload];
        }
        continue;
      }
      if (opcode === OPCODE.CONTINUATION) {
        this._fragments.push(payload);
        if (fin) {
          const full = Buffer.concat(this._fragments);
          const op = this._fragmentOpcode ?? OPCODE.BINARY;
          this._fragments = [];
          this._fragmentOpcode = null;
          this._emitMessage(op, full);
        }
        continue;
      }
      throw new Error(`unknown opcode ${opcode}`);
    }
  }

  _emitMessage(opcode, payload) {
    if (opcode === OPCODE.TEXT) {
      this.emit('message', payload.toString('utf8'), false);
    } else {
      this.emit('message', payload, true);
    }
  }

  _sendFrame(opcode, payload) {
    if (this.readyState !== 'open' || this.socket.destroyed) return;
    const len = payload.length;
    let header;
    if (len < 126) {
      header = Buffer.allocUnsafe(2);
      header[1] = len;
    } else if (len < 65536) {
      header = Buffer.allocUnsafe(4);
      header[1] = 126;
      header.writeUInt16BE(len, 2);
    } else {
      header = Buffer.allocUnsafe(10);
      header[1] = 127;
      header.writeBigUInt64BE(BigInt(len), 2);
    }
    header[0] = 0x80 | opcode;
    this.bytesOut += header.length + len;
    this.socket.write(Buffer.concat([header, payload]));
  }

  send(data) {
    const isBinary = Buffer.isBuffer(data) || data instanceof Uint8Array;
    const payload = isBinary ? Buffer.from(data) : Buffer.from(String(data), 'utf8');
    this._sendFrame(isBinary ? OPCODE.BINARY : OPCODE.TEXT, payload);
  }

  sendJSON(obj) {
    this.send(JSON.stringify(obj));
  }

  ping() {
    this._sendFrame(OPCODE.PING, Buffer.alloc(0));
  }

  close(code = 1000, reason = '') {
    if (this.readyState === 'closed') return;
    this.readyState = 'closing';
    const reasonBuf = Buffer.from(reason, 'utf8');
    const payload = Buffer.allocUnsafe(2 + reasonBuf.length);
    payload.writeUInt16BE(code, 0);
    reasonBuf.copy(payload, 2);
    try {
      this._sendFrame(OPCODE.CLOSE, payload);
    } catch {
      /* ignore */
    }
    setTimeout(() => {
      try {
        this.socket.end();
      } catch {
        /* ignore */
      }
      this._finish();
    }, 30).unref?.();
  }
}

export class WebSocketServer extends EventEmitter {
  /**
   * @param {import('node:http').Server} httpServer
   * @param {{path?: string}} [options]
   */
  constructor(httpServer, options = {}) {
    super();
    this.path = options.path ?? '/ws';
    this.clients = new Set();
    httpServer.on('upgrade', (req, socket, head) => {
      const url = new URL(req.url, 'http://localhost');
      if (url.pathname !== this.path) {
        socket.write('HTTP/1.1 404 Not Found\r\n\r\n');
        socket.destroy();
        return;
      }
      const key = req.headers['sec-websocket-key'];
      if (!key) {
        socket.write('HTTP/1.1 400 Bad Request\r\n\r\n');
        socket.destroy();
        return;
      }
      const accept = createHash('sha1').update(key + GUID).digest('base64');
      socket.write(
        'HTTP/1.1 101 Switching Protocols\r\n' +
          'Upgrade: websocket\r\n' +
          'Connection: Upgrade\r\n' +
          `Sec-WebSocket-Accept: ${accept}\r\n\r\n`
      );
      const conn = new WebSocketConnection(socket);
      if (head && head.length) socket.unshift(head);
      this.clients.add(conn);
      conn.on('close', () => {
        this.clients.delete(conn);
        this.emit('close', conn);
      });
      this.emit('connection', conn, req);
    });
  }

  broadcast(data, filter) {
    for (const client of this.clients) {
      if (client.readyState !== 'open') continue;
      if (filter && !filter(client)) continue;
      client.send(data);
    }
  }

  close() {
    for (const c of this.clients) c.close(1001, 'server shutdown');
    this.clients.clear();
  }
}
