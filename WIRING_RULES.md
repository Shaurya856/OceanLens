# Frontend–Backend Wiring Rules

These rules govern every page in `frontend/`. Follow them exactly when
converting buttons, drop zones, or forms from dummy UI to live API calls.

---

## 1. API Base

The frontend is served at `/ui/` by the same FastAPI process.  
All API paths are **root-relative** (`/enhance`, `/ws/…`, etc.) — no hostname needed.

```js
// Correct
fetch('/enhance', { method: 'POST', body: fd })

// WebSocket — must include host explicitly
const ws = new WebSocket(
    (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/' + jobId
);
```

---

## 2. Job ID

**Always generate the job ID on the client** with `crypto.randomUUID()`.  
This lets you open the WebSocket *before* posting the job so no events are missed.

```js
const jobId = crypto.randomUUID();
const ws = connectJob(jobId, handlers);   // open WS first
const res = await fetch('/enhance', …);   // then POST
```

---

## 3. WebSocket Flow

1. Call `connectJob(jobId, handlers)` from `app.js` **before** the POST.
2. Post the job. The backend schedules it as an async task — the WS will be
   ready before the first event fires.
3. `connectJob` dispatches on the `status` field:

| `status` value     | Handler called        | Endpoint(s) |
|--------------------|-----------------------|-------------|
| `extracting_frames`| `onProgress`          | video pipelines |
| `enhancing`        | `onProgress`          | `/pipeline/video/enhance` |
| `processing`       | `onProgress`          | detect pipelines |
| `completed`        | `onCompleted`         | `/enhance` (per-image) |
| `frame_done`       | `onFrameDone`         | video pipelines |
| `image_done`       | `onImageDone`         | `/pipeline/image/enhance-detect` |
| `frame_failed`     | `onFrameFailed`       | video pipelines |
| `image_failed`     | `onImageFailed`       | image pipeline |
| `job_done`         | `onDone` + close WS   | all endpoints |
| `failed`           | `onError` + close WS  | all endpoints |

4. Store the WS reference so Cancel buttons can call `ws.close()`.

---

## 4. Enhancement Technique Serialisation

Only include techniques whose checkbox is **checked**.  
Always send `mode=custom` to `/enhance` — it accepts any number of techniques ≥ 1.

### UI element → backend key

| UI label          | `techniques` array key | Params sent |
|-------------------|------------------------|-------------|
| Denoise           | `denoise`              | `{ h: <sigma slider value> }` |
| CLAHE             | `clahe`                | `{ clip_limit: <float>, tile_size: <int> }` |
| Gamma Correction  | `gamma_correction`     | `{ gamma: <float> }` |
| White Balance     | `white_balance`        | *(no params)* |
| Dehaze            | `dehaze`               | `{ omega: <strength slider value> }` |
| Retinex           | `retinex`              | `{ sigmas: [15, 80, 250] }` (parse comma input) |
| Super Resolution  | `superres`             | `{ scale: 2 | 4 | 8 }` (from active btn-scale) |

### CLAHE tile size parsing

The tile input holds a string like `"8x8"` or `"8"`. Always `parseInt()` — the
backend expects an integer for the NxN grid.

### Retinex scales parsing

```js
const sigmas = document.getElementById('retinex-scales').value
    .split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
// Fallback to [15, 80, 250] if input is empty or invalid
```

---

## 5. HTML ID Conventions

Every interactive element must have a stable `id`. Naming scheme:

| Purpose | Pattern | Example |
|---------|---------|---------|
| File input | `file-input` | `id="file-input"` |
| Drop zone | `drop-zone` | `id="drop-zone"` |
| Primary action | `run-btn` or `start-btn` | `id="run-btn"` |
| Cancel | `cancel-btn` | `id="cancel-btn"` |
| Export CSV | `export-csv-btn` | `id="export-csv-btn"` |
| Download All | `download-all-btn` | `id="download-all-btn"` |
| Progress fill bar | `progress-fill` | `id="progress-fill"` |
| Progress % label | `progress-pct` | `id="progress-pct"` |
| Results container | `results-container` | `id="results-container"` |
| Log terminal | `log-terminal` | `id="log-terminal"` |
| Footer status dot | `status-dot` | `id="status-dot"` |
| Footer status text | `status-text` | `id="status-text"` |
| Footer job display | `job-display` | `id="job-display"` |

**Page-specific element prefixes** match the existing slider ID prefix convention:

| Page | Prefix | Example |
|------|--------|---------|
| image-enhanced | *(none)* | `id="denoise-check"` |
| image-species-enhanced | `se-` | `id="se-denoise-check"` |
| video-enhanced | `v-` | `id="v-denoise-check"` |
| video-species | `vs-` | `id="vs-denoise-check"` |

---

## 6. Download and Export

Use helpers from `app.js`:

```js
import { triggerDownload, exportCSV } from '/ui/app.js';

// Single file download
triggerDownload('/download/' + jobId + '/' + filename, filename);

// Extracted frames (from /video/ingest)
triggerDownload('/frames/' + jobId + '/' + filename, filename);

// CSV export
exportCSV(rows, 'results.csv');  // rows = array of flat objects
```

---

## 7. Error Handling

- **No file selected**: show an `alert()` and `return` — do not POST.
- **No techniques selected** (for `/enhance`): show `alert()` and `return`.
- **Fetch error**: catch and show inline error in the log terminal.
- **WS `failed` event**: log the `error` field, reset button state.
- Never let an unhandled error break the page.

---

## 8. Button State Management

Disable the action button while a job runs; re-enable on `job_done` or `failed`.

```js
runBtn.disabled = true;
runBtn.textContent = 'Running…';
// … on job_done or failed:
runBtn.disabled = false;
runBtn.textContent = 'Run Detection';
```

---

## 9. Status Bar

Update the footer on every state change:

```js
import { setStatus } from '/ui/app.js';

setStatus('busy', 'Enhancing…', jobId);   // dot turns amber
setStatus('live', 'Streaming', jobId);    // dot turns green
setStatus('idle', 'Ready', null);         // dot returns to grey
```

---

## 10. Script Tag Convention

Every page uses a **single `<script type="module">`** block placed just before
`</body>`. It:
1. Imports shared utilities from `/ui/app.js`
2. Keeps all existing slider/toggle event listeners (do not delete them)
3. Adds drop zone wiring
4. Adds submit/cancel/export/download handlers

No inline `onclick` attributes — all events attached via `addEventListener`.
