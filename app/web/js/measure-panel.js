/**
 * 精确测量面板
 *
 * 对应两个业务需求：
 *   ① 民警可以测定房间的准确数据并上传 → 使建模更精确
 *   ② 在粗略建模文件里标出「哪里需要精确数据」，民警测量后直接填写
 *
 * 工作流：
 *   1) 技术人员在粗模上用「📏 尺寸标注」工具点出两个端点
 *      → 生成一条 dimension 标注（measured = null，即「待实测」）
 *   2) 现场民警打开「待测清单」，对照屏幕上的红点/红线位置，用卷尺量出真实距离并填写
 *   3) 点「按实测量校正比例尺」→ 后端最小二乘反算 scaleToMeters
 *      → 整个模型与所有测距立刻按真实尺寸显示
 *
 * 另外支持导出 CSV，便于打印带到现场（现场可能没有设备）。
 */

import { ensureDialogClosable, openDialogModal } from './ui.js';

export class MeasurePanel {
  constructor({ currentProjectId, getProject, operator, onCalibrated, toast, onUpdateAnnotation } = {}) {
    this.currentProjectId = currentProjectId ?? (() => null);
    this.getProject = getProject ?? (() => null);
    this.operator = operator ?? (() => '未署名');
    this.onCalibrated = onCalibrated ?? (async () => {});
    this.toast = toast ?? (() => {});
    this.onUpdateAnnotation = onUpdateAnnotation ?? (async () => {});
    /** 表单里正在编辑的值：id -> string */
    this.draft = new Map();
    this.data = null;
  }

  init() {
    const $ = (id) => {
      const el = document.getElementById(id);
      if (!el) console.error(`[measure] 缺少元素 #${id}`);
      return el;
    };
    this.el = {
      badge: $('measureBadge'),
      listBtn: $('btnMeasureList'),
      recalBtn: $('btnRecalibrate'),
      scaleHint: $('scaleHint'),
      dlg: $('dlgMeasure'),
      table: $('measureTable'),
      done: $('measureDone'),
      summary: $('measureSummary'),
      saveBtn: $('btnMeasureSave'),
      exportBtn: $('btnMeasureExport'),
    };
    this.el.listBtn?.addEventListener('click', () => this.open());
    this.el.recalBtn?.addEventListener('click', () => this.recalibrate());
    this.el.saveBtn?.addEventListener('click', () => this.saveAll());
    this.el.exportBtn?.addEventListener('click', () => this.exportCsv());
    // 保证「关闭」按钮在**任何内核**里都能关掉窗口（不依赖 <form method="dialog">）
    ensureDialogClosable(this.el.dlg);
    this.refresh();
    return this;
  }

  /** 拉取测量清单并更新左侧徽标/提示 */
  async refresh() {
    const pid = this.currentProjectId();
    if (!pid) {
      if (this.el.badge) this.el.badge.textContent = '0';
      if (this.el.scaleHint) this.el.scaleHint.textContent = '';
      return;
    }
    try {
      const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/measurements`);
      if (!res.ok) return;
      const data = await res.json();
      this.data = data;
      const pending = (data.pending ?? []).length;
      const measured = (data.measured ?? []).length;
      if (this.el.badge) {
        this.el.badge.textContent = String(pending);
        this.el.badge.classList.toggle('warn', pending > 0);
      }
      if (this.el.scaleHint) {
        const scale = Number(data.scaleToMeters ?? 1);
        const parts = [];
        if (pending) parts.push(`<b class="warn">${pending}</b> 项待实测`);
        if (measured) parts.push(`${measured} 项已实测`);
        if (data.calibratedAt) {
          const conf = { high: '高', medium: '中', low: '偏低' }[data.calibration?.confidence] ?? '';
          parts.push(`比例尺 ${scale.toFixed(4)}${conf ? `（可信度${conf}）` : ''}`);
        } else {
          parts.push('比例尺未校正（当前 1.0000）');
        }
        this.el.scaleHint.innerHTML = parts.join(' · ');
      }
      return data;
    } catch {
      return null;
    }
  }

  async open() {
    const pid = this.currentProjectId();
    if (!pid) {
      this.toast('请先选择案件', { error: true });
      return;
    }
    const data = await this.refresh();
    if (!data) {
      this.toast('读取测量清单失败', { error: true });
      return;
    }
    this.render(data);
    this.el.dlg && openDialogModal(this.el.dlg);
  }

  render(data) {
    const pending = data?.pending ?? [];
    const measured = data?.measured ?? [];
    const scale = Number(data?.scaleToMeters ?? 1) || 1;

    if (this.el.summary) {
      this.el.summary.innerHTML =
        `待实测 <b class="${pending.length ? 'warn' : ''}">${pending.length}</b> 项 · ` +
        `已实测 <b class="ok">${measured.length}</b> 项 · 当前比例尺 <b>${scale.toFixed(4)}</b>`;
    }

    // 待测表单
    if (!pending.length) {
      this.el.table.innerHTML =
        '<div class="measure-empty">没有待实测项。<br>请用工具栏的「📏 尺寸标注」在三维现场点出需要精确测量的两个端点。</div>';
    } else {
      this.el.table.innerHTML = pending
        .map((p, i) => {
          const v = this.draft.get(p.id) ?? '';
          const prio = { high: '必测', normal: '常规', low: '可缓' }[p.priority] ?? '常规';
          return `<div class="measure-row" data-id="${p.id}">
            <div class="measure-info">
              <div class="measure-name">
                <span class="idx">${i + 1}</span>
                <b>${this._esc(p.label || `尺寸 ${i + 1}`)}</b>
                <span class="prio ${p.priority ?? 'normal'}">${prio}</span>
              </div>
              <div class="measure-model">模型侧：<b>${(Number(p.modelDistance) * scale).toFixed(4)} m</b>
                ${p.note ? `<span class="note">${this._esc(p.note)}</span>` : ''}
              </div>
            </div>
            <label class="measure-input">
              <input type="number" step="0.001" min="0" inputmode="decimal"
                     placeholder="实测 m" data-measure-input="${p.id}"
                     value="${this._esc(v)}" />
              <span>m</span>
            </label>
          </div>`;
        })
        .join('');
      for (const input of this.el.table.querySelectorAll('[data-measure-input]')) {
        input.addEventListener('input', () => this.draft.set(input.dataset.measureInput, input.value));
      }
    }

    // 已测列表
    if (measured.length) {
      this.el.done.classList.remove('hidden');
      this.el.done.innerHTML =
        '<div class="measure-done-head">已实测（可修改后重新校正）</div>' +
        measured
          .map((m) => {
            const err = Number(m.measured) - Number(m.modelDistance) * scale;
            const ok = Math.abs(err) < 0.005;
            return `<div class="measure-row done" data-id="${m.id}">
              <div class="measure-info">
                <div class="measure-name"><b>${this._esc(m.label || '尺寸')}</b></div>
                <div class="measure-model">
                  实测 <b class="ok">${Number(m.measured).toFixed(4)} m</b>
                  · 模型 ${(Number(m.modelDistance) * scale).toFixed(4)} m
                  · 偏差 <b class="${ok ? 'ok' : 'warn'}">${err >= 0 ? '+' : ''}${err.toFixed(4)} m</b>
                  ${m.measuredBy ? `<span class="note">由 ${this._esc(m.measuredBy)} 测量</span>` : ''}
                </div>
              </div>
              <label class="measure-input">
                <input type="number" step="0.001" min="0" inputmode="decimal"
                       data-measure-input="${m.id}" value="${Number(m.measured)}" />
                <span>m</span>
              </label>
            </div>`;
          })
          .join('');
      for (const input of this.el.done.querySelectorAll('[data-measure-input]')) {
        input.addEventListener('input', () => this.draft.set(input.dataset.measureInput, input.value));
      }
    } else {
      this.el.done.classList.add('hidden');
      this.el.done.innerHTML = '';
    }

    if (this.el.saveBtn) this.el.saveBtn.disabled = pending.length === 0 && measured.length === 0;
    if (this.el.exportBtn) this.el.exportBtn.disabled = pending.length === 0;
  }

  _esc(s) {
    return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  }

  /** 把所有填写的实测值提交到后端 */
  async saveAll() {
    const pid = this.currentProjectId();
    if (!pid) return;
    if (!this.draft.size) {
      this.toast('没有需要保存的修改');
      return;
    }
    let ok = 0;
    let bad = 0;
    for (const [id, raw] of this.draft) {
      const v = Number(raw);
      if (!Number.isFinite(v) || v <= 0) {
        if (String(raw).trim() !== '') bad++;
        continue;
      }
      try {
        await this.onUpdateAnnotation(id, { measured: v, measuredBy: this.operator() });
        ok++;
      } catch {
        bad++;
      }
    }
    this.draft.clear();
    const data = await this.refresh();
    if (data) this.render(data);
    if (bad) this.toast(`保存了 ${ok} 项，${bad} 项数值无效已跳过`, { error: true });
    else this.toast(`已保存 ${ok} 项实测数据。可点「按实测量校正」让模型按真实尺寸显示`);
  }

  /** 触发比例尺校正 */
  async recalibrate() {
    const pid = this.currentProjectId();
    if (!pid) return;
    try {
      const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/calibrate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ operator: this.operator() }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) throw new Error(data.error ?? `校正失败 ${res.status}`);
      await this.onCalibrated(data);
      const refreshed = await this.refresh();
      if (refreshed) this.render(refreshed);
      const conf = { high: '高', medium: '中', low: '偏低（建议复核标注点是否点准）' }[data.confidence] ?? data.confidence;
      this.toast(`比例尺已校正为 ${Number(data.scaleToMeters).toFixed(4)}（用了 ${data.count} 条实测数据，可信度${conf}）`);
    } catch (err) {
      this.toast(err.message, { error: true });
    }
  }

  /** 导出 CSV：现场可能没有设备，打印出来对着量更实用 */
  exportCsv() {
    const pending = this.data?.pending ?? [];
    if (!pending.length) {
      this.toast('没有待实测项可导出');
      return;
    }
    const scale = Number(this.data?.scaleToMeters ?? 1) || 1;
    const rows = [
      ['序号', '名称', '模型侧距离(m)', '实测距离(m)', '优先级', '备注'],
      ...pending.map((p, i) => [
        i + 1,
        p.label ?? `尺寸 ${i + 1}`,
        (Number(p.modelDistance) * scale).toFixed(4),
        '',
        { high: '必测', normal: '常规', low: '可缓' }[p.priority] ?? '常规',
        p.note ?? '',
      ]),
    ];
    // 带 BOM，Excel 打开中文不乱码
    const csv =
      '\ufeff' +
      rows
        .map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))
        .join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const caseName = this.getProject?.()?.name ?? '现场';
    a.download = `${caseName}-精确测量清单.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    this.toast('已导出测量清单 CSV，可打印带去现场');
  }

  /** 切案件时同步 */
  onProjectChanged() {
    this.draft.clear();
    this.data = null;
    this.refresh();
  }
}
