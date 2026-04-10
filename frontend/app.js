/**
 * app.js — shared OceanLens AI frontend utilities
 *
 * Import in every page with:
 *   import { connectJob, initDropZone, … } from '/ui/app.js';
 *
 * All functions follow the conventions in WIRING_RULES.md.
 */

// ── WebSocket ─────────────────────────────────────────────────────────────────

/**
 * Open a WebSocket for the given jobId and dispatch status events to handlers.
 *
 * handlers:
 *   onProgress(msg)     — extracting_frames | enhancing | processing
 *   onCompleted(msg)    — completed  (single-stage /enhance per-image event)
 *   onFrameDone(msg)    — frame_done
 *   onImageDone(msg)    — image_done
 *   onFrameFailed(msg)  — frame_failed
 *   onImageFailed(msg)  — image_failed
 *   onDone(msg)         — job_done  (WS closed automatically after)
 *   onError(msg)        — failed    (WS closed automatically after)
 *   onClose()           — WS closed for any reason
 *
 * Returns the WebSocket instance so callers can call ws.close() to cancel.
 */
export function connectJob(jobId, handlers = {}) {
    const proto  = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws     = new WebSocket(`${proto}//${location.host}/ws/${jobId}`);

    ws.onmessage = (event) => {
        let msg;
        try { msg = JSON.parse(event.data); } catch { return; }

        switch (msg.status) {
            case 'extracting_frames': handlers.onProgress?.(msg);  break;
            case 'enhancing':         handlers.onProgress?.(msg);  break;
            case 'processing':        handlers.onProgress?.(msg);  break;
            case 'completed':         handlers.onCompleted?.(msg); break;
            case 'frame_done':        handlers.onFrameDone?.(msg); break;
            case 'image_done':        handlers.onImageDone?.(msg); break;
            case 'frame_failed':      handlers.onFrameFailed?.(msg); break;
            case 'image_failed':      handlers.onImageFailed?.(msg); break;
            case 'job_done':
                handlers.onDone?.(msg);
                ws.close();
                break;
            case 'failed':
                handlers.onError?.(msg);
                ws.close();
                break;
        }
    };

    ws.onerror = () => handlers.onError?.({ error: 'WebSocket connection error' });
    ws.onclose = () => handlers.onClose?.();

    return ws;
}

// ── Drop Zone ─────────────────────────────────────────────────────────────────

/**
 * Wire up a drop zone label + hidden file input.
 *
 * @param {string}   dropZoneId  id of the <label class="drop-zone"> element
 * @param {string}   fileInputId id of the <input type="file"> inside the zone
 * @param {Function} onFiles     called with Array<File> whenever files are chosen
 */
export function initDropZone(dropZoneId, fileInputId, onFiles) {
    const zone  = document.getElementById(dropZoneId);
    const input = document.getElementById(fileInputId);
    if (!zone || !input) return;

    input.addEventListener('change', () => {
        if (input.files.length) onFiles(Array.from(input.files));
    });

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('ring-2', 'ring-blue-400');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('ring-2', 'ring-blue-400');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('ring-2', 'ring-blue-400');
        const files = Array.from(e.dataTransfer.files);
        if (files.length) onFiles(files);
    });
}

// ── Progress Bar ──────────────────────────────────────────────────────────────

/**
 * Update a progress bar fill div and its companion percentage label.
 *
 * @param {string} fillId  id of the <div class="progress-fill"> element
 * @param {string} pctId   id of the element showing the "N%" text
 * @param {number} done    completed items
 * @param {number} total   total items
 */
export function updateProgress(fillId, pctId, done, total) {
    const pct   = total > 0 ? Math.round((done / total) * 100) : 0;
    const fill  = document.getElementById(fillId);
    const label = document.getElementById(pctId);
    if (fill)  fill.style.width = pct + '%';
    if (label) label.textContent = pct + '%';
}

// ── Status Bar ────────────────────────────────────────────────────────────────

/**
 * Update the footer status bar.
 *
 * @param {'idle'|'busy'|'live'} state  controls the dot colour
 * @param {string}               text   status message
 * @param {string|null}          jobId  job UUID to display (or null to clear)
 */
export function setStatus(state, text, jobId = null) {
    const dot     = document.getElementById('status-dot');
    const label   = document.getElementById('status-text');
    const jobDisp = document.getElementById('job-display');

    if (dot) {
        dot.classList.remove('live', 'busy');
        if (state === 'live') dot.classList.add('live');
        if (state === 'busy') dot.classList.add('busy');
    }
    if (label)   label.textContent = text;
    if (jobDisp) jobDisp.textContent = jobId ?? '—';
}

// ── Log Terminal ──────────────────────────────────────────────────────────────

/**
 * Append a timestamped line to a log terminal element.
 *
 * @param {string} terminalId  id of the log terminal container
 * @param {string} message     text to append
 */
export function log(terminalId, message) {
    const terminal = document.getElementById(terminalId);
    if (!terminal) return;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const p    = document.createElement('p');
    p.textContent = `[${time}] ${message}`;
    terminal.appendChild(p);
    terminal.scrollTop = terminal.scrollHeight;
}

// ── Download Helpers ──────────────────────────────────────────────────────────

/**
 * Trigger a browser file download for a URL.
 *
 * @param {string} url      absolute or root-relative URL
 * @param {string} filename suggested download filename
 */
export function triggerDownload(url, filename) {
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ── CSV Export ────────────────────────────────────────────────────────────────

/**
 * Export an array of flat objects as a CSV file download.
 *
 * @param {Object[]} rows      array of plain objects (all with same keys)
 * @param {string}   filename  download filename (e.g. 'results.csv')
 */
export function exportCSV(rows, filename) {
    if (!rows || !rows.length) return;
    const keys  = Object.keys(rows[0]);
    const lines = [
        keys.join(','),
        ...rows.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(',')),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    triggerDownload(URL.createObjectURL(blob), filename);
}

// ── Detection Card Builder ────────────────────────────────────────────────────

/**
 * Build a collapsible detection card with canvas bbox overlay.
 *
 * @param {string|null} imgSrc     Object URL (or server URL) for the image.
 *                                 Pass null to omit the viewer.
 * @param {string}      filename   Filename shown in the header.
 * @param {Array}       detections Detection array from results JSON.
 *                                 Each item: { bbox:{x1,y1,x2,y2}, confidence,
 *                                              taxonomy:{species,...}, is_novel }
 * @returns {HTMLElement} A .det-card element ready to append.
 */
export function buildDetCard(imgSrc, filename, detections = []) {
    const detCount   = detections.length;
    const novelCount = detections.filter(d => d.is_novel).length;

    const card = document.createElement('div');
    card.className = 'det-card';

    // ── Header ────────────────────────────────────────────────────────────────
    const header = document.createElement('div');
    header.className = 'det-card-header';
    const novelBadge = novelCount > 0
        ? `<span class="badge" style="background:#FEF3C7;border-color:#FDE68A;color:#92400E">${novelCount} novel</span>`
        : '';
    header.innerHTML = `
        <span class="det-card-filename" title="${filename}">${filename}</span>
        <div class="det-card-badges">
            <span class="badge">${detCount} det</span>${novelBadge}
        </div>
        <button class="det-card-toggle" aria-label="Toggle">
            <span class="material-symbols-outlined">expand_more</span>
        </button>`;

    // ── Body ──────────────────────────────────────────────────────────────────
    const body  = document.createElement('div');
    body.className = 'det-card-body';
    const inner = document.createElement('div');
    inner.className = 'det-card-body-inner';

    if (imgSrc) {
        const viewer = document.createElement('div');
        viewer.className = 'det-card-viewer';

        const img    = document.createElement('img');
        img.className = 'det-card-img';
        img.alt       = filename;
        img.src       = imgSrc;

        const canvas = document.createElement('canvas');
        canvas.className = 'det-card-canvas';
        viewer.append(img, canvas);
        inner.appendChild(viewer);

        const draw = () => {
            canvas.width  = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            const lw  = Math.max(2, Math.round(canvas.width / 400));
            const fs  = Math.max(11, Math.round(canvas.width / 70));

            detections.forEach(d => {
                const { x1, y1, x2, y2 } = d.bbox;
                const color = d.is_novel ? '#F59E0B' : '#3B82F6';
                const label = `${d.taxonomy?.species ?? '?'}  ${(d.confidence * 100).toFixed(0)}%`;

                ctx.strokeStyle = color;
                ctx.lineWidth   = lw;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                ctx.font = `${fs}px ui-monospace, monospace`;
                const tw  = ctx.measureText(label).width;
                const pad = 4;
                const lh  = fs + pad * 2;
                ctx.fillStyle = color;
                ctx.fillRect(x1, y1 - lh, tw + pad * 2, lh);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, x1 + pad, y1 - pad);
            });
        };

        if (img.complete && img.naturalWidth > 0) draw();
        else img.addEventListener('load', draw, { once: true });
    }

    // Per-detection rows
    const detList = document.createElement('div');
    detList.className = 'det-card-detections';
    if (!detections.length) {
        detList.innerHTML = '<div class="det-card-no-det">No detections</div>';
    } else {
        detections.forEach(d => {
            const row = document.createElement('div');
            row.className = 'det-card-det-row';
            row.innerHTML = `
                <span class="det-card-det-species">${d.taxonomy?.species ?? '—'}</span>
                <span class="det-card-det-conf">${(d.confidence * 100).toFixed(1)}%</span>
                ${d.is_novel ? '<span class="det-card-det-novel">Novel</span>' : ''}`;
            detList.appendChild(row);
        });
    }
    inner.appendChild(detList);
    body.appendChild(inner);
    card.append(header, body);

    header.addEventListener('click', () => card.classList.toggle('open'));
    return card;
}

// ── Enhancement Param Collector ───────────────────────────────────────────────

/**
 * Read technique checkboxes and sliders for one page and return the
 * { techniques, params } object ready to JSON-serialise into FormData.
 *
 * @param {Object} ids  map of element IDs for this page's controls:
 *   {
 *     denoiseCheck, denoiseSlider,
 *     claheCheck, claheClip, claheTile,
 *     gammaCheck, gammaSlider,
 *     wbCheck,
 *     dehazeCheck, dehazeSlider,
 *     retinexCheck, retinexScales,
 *     srCheck, srScaleGroup,
 *   }
 *
 * Only checked techniques are included.  Empty `ids` fields are skipped.
 */
export function collectEnhancement(ids) {
    const techniques = [];
    const params     = {};

    const checked = (id) => id && document.getElementById(id)?.checked;
    const val     = (id) => document.getElementById(id)?.value ?? '';

    if (checked(ids.denoiseCheck)) {
        techniques.push('denoise');
        params.denoise = { h: parseFloat(val(ids.denoiseSlider)) || 10 };
    }

    if (checked(ids.claheCheck)) {
        techniques.push('clahe');
        params.clahe = {
            clip_limit: parseFloat(val(ids.claheClip)) || 3.0,
            tile_size:  parseInt(val(ids.claheTile))   || 8,
        };
    }

    if (checked(ids.gammaCheck)) {
        techniques.push('gamma_correction');
        params.gamma_correction = { gamma: parseFloat(val(ids.gammaSlider)) || 1.2 };
    }

    if (checked(ids.wbCheck)) {
        techniques.push('white_balance');
    }

    if (checked(ids.dehazeCheck)) {
        techniques.push('dehaze');
        params.dehaze = { omega: parseFloat(val(ids.dehazeSlider)) || 0.95 };
    }

    if (checked(ids.retinexCheck)) {
        techniques.push('retinex');
        const raw    = val(ids.retinexScales);
        const sigmas = raw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
        params.retinex = { sigmas: sigmas.length ? sigmas : [15, 80, 250] };
    }

    if (checked(ids.srCheck)) {
        techniques.push('superres');
        const group     = ids.srScaleGroup ? document.getElementById(ids.srScaleGroup) : null;
        const activeBtn = group
            ? group.querySelector('.btn-scale.active')
            : document.querySelector('.btn-scale.active');
        const scale     = activeBtn ? parseInt(activeBtn.textContent) : 2;
        params.superres = { scale: isNaN(scale) ? 2 : scale };
    }

    return { techniques, params };
}
