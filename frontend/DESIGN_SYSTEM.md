# OceanLens Design System

Single source of truth for component classes. All defined in `styles.css`.
Use this file as context instead of reading individual HTML pages.

---

## Layout Chrome
Fixed shell present on all 6 pages. Only `.active` on nav-item changes between pages.

| Class | Element | Notes |
|---|---|---|
| `.app-sidebar` | `<aside>` | Fixed 220px left panel |
| `.sidebar-brand` | `<div>` | Top section of sidebar; direct child `<h1>` gets Geist font via CSS |
| `.app-header` | `<header>` | Fixed top bar, h=3.5rem, left=220px |
| `.app-main` | `<main>` | ml-220px, pt-3.5rem, pb-3rem. Add `px-8` for pages with full-width content |
| `.app-footer` | `<footer>` | Fixed bottom bar, h=2.5rem, left=220px. Add `justify-between` or `gap-6` as needed |

---

## Navigation

| Class | Modifier | Notes |
|---|---|---|
| `.nav-item` | — | Sidebar link; gray text, hover darkens |
| `.nav-item` | `.active` | Dark bg, right border accent, bold text |

---

## Cards

| Class | Notes |
|---|---|
| `.card` | White box, 1px gray-200 border, 2px border-radius |
| `.card-body` | Padding: 1rem (16px) |
| `.card-body-md` | Padding: 1.25rem (20px) |
| `.card-header` | 0.75rem 1rem padding, border-bottom, flex justify-between |
| `.card-title` | 11px bold uppercase — for card section headings with icons |

Usage: `<div class="card card-body">` or `<div class="card overflow-hidden"><div class="card-header">...</div><div class="card-body">...</div></div>`

---

## Buttons

| Class | Notes |
|---|---|
| `.btn-primary` | Dark fill, 10px uppercase. Add `w-full` for full-width. Has icon slot via gap. |
| `.btn-secondary` | White outlined, 12px normal case. Inline with icon slot. |
| `.btn-icon` | 28×28px square icon button |
| `.btn-scale` | Flex-1 toggle button for scale selectors (SR, etc.) |
| `.btn-scale.active` | Dark fill active state |

---

## Forms

| Class | Notes |
|---|---|
| `.form-input` | Full-width text/number field, 14px, standard padding |
| `.form-input-sm` | Compact field, gray-50 bg, 12px — for inline config fields |
| `.form-select` | Full-width select, gray-50 bg, 12px |
| `.checkbox` | Standardized 16×16px checkbox, accent-gray-900 |
| `.field-row` | flex justify-between for slider label + value pairs |
| `.field-label` | 11px medium gray-700 — field label text |
| `.field-value` | 10px mono gray-900 — live value readout |

---

## Header Search

| Class | Element | Notes |
|---|---|---|
| `.header-search-wrap` | `<div>` | Position relative wrapper |
| `.header-search-icon` | `<span class="material-symbols-outlined ...">` | Absolutely positioned search icon |
| `.header-search` | `<input>` | 14rem wide search field |
| `.header-btn` | `<button>` | Icon button for notifications/settings/account |

---

## Upload / Drop Zone

| Class | Notes |
|---|---|
| `.drop-zone` | Applied to `<label>`. Dashed border, gray-50 bg, 2.5rem padding. |
| `.drop-zone-icon` | Applied to icon `<span>`. Forces 2.5rem size, gray-300 color. |
| `.drop-zone h3` | 14px semibold title (styled via CSS child selector) |
| `.drop-zone p` | 12px subtitle (styled via CSS child selector) |

---

## Data Display

| Class | Notes |
|---|---|
| `.data-table` | Applied to `<table>`. No inline th/td classes needed. |
| `.badge` | 10px uppercase pill: gray-100 bg, gray-200 border |
| `.empty-state` | Centered column for empty content areas |
| `.section-label` | 10px uppercase gray-400 — field captions, card labels |

---

## Status & Feedback

| Class | Notes |
|---|---|
| `.status-dot` | 7px circle, gray by default |
| `.status-dot.live` | Green (#22C55E) |
| `.status-dot.busy` | Amber (#F59E0B) |
| `.status-text` | 10px uppercase bold gray-600 — "WebSocket Idle" etc. |
| `.progress-track` | Progress bar container |
| `.progress-fill` | Progress bar fill — set `width` via inline style or JS |
| `.step-circle` | Stepper step number circle |
| `.step-circle.current` | Dark border + text |
| `.step-circle.done` | Dark fill |
| `.step-line` | Flex-1 horizontal connector between steps |
| `.log-terminal` | Dark monospace log box, 7rem height |

---

## Technique Cards

`.technique-card` — white card with border for enhancement config blocks. Use `space-y-3` inside.

Structure:
```html
<div class="technique-card space-y-3">
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-2">
      <span class="material-symbols-outlined text-gray-400 text-sm">{icon}</span>
      <span class="text-xs font-bold text-gray-900">{Name}</span>
    </div>
    <input type="checkbox" class="checkbox"/>
  </div>
  <!-- controls -->
</div>
```

---

## Typography

| Class | Size | Font | Notes |
|---|---|---|---|
| `.page-title` | 20px | Geist bold | h2 page headings |
| `.page-subtitle` | 14px | Inter | Subtitle below page-title, gray-600 |
| `.card-title` | 11px | Inter bold uppercase | Card section headings |
| `.section-label` | 10px | Inter bold uppercase | Field captions, gray-400 |
| `.field-label` | 11px | Inter medium | Input labels |
| `.field-value` | 10px | Mono | Live slider values |

---

## Spacing Conventions

- Card inner padding: `card-body` (1rem) or `card-body-md` (1.25rem)
- Section gaps: `gap-6` between major grid areas, `gap-4` for card grids, `gap-3` inside cards
- Page content max-width: `max-w-[1100px]` (standard), `max-w-[900px]` (narrow pages)
- Top padding inside main: `pt-8`

---

## Unified Enhancement + Upload Layout

Pages that perform enhancement share a consistent two-column structure:

| Column | Width | Contents |
|---|---|---|
| Left | 4/12 | Upload drop zone → Enhancement Pipeline card (technique-cards) → Run button |
| Right | 8/12 | Preview / Detection Feed + Download button(s) |

**Enhancement Techniques** (all 7 available on every enhancement page):

| Technique | Controls | Default |
|---|---|---|
| Denoise | Sigma Level slider (0–50) | On |
| CLAHE | Clip Limit + Tile Size text fields | On |
| Gamma Correction | Gamma slider (0.5–3.0), toggle-disabled | Off |
| White Balance | Method select (Gray World / White Patch / Learning-Based) | Off |
| Dehaze | Strength slider (0–1) | On |
| Retinex | Method select + Scales text, toggle-disabled | Off |
| Super Resolution | Scale toggle buttons (2× / 4× / 8×), toggle-disabled | Off |

Super Resolution is off by default on video pages due to per-frame processing cost, but is always available.

Status bars use `Idle` / `Ready` — no WebSocket-specific text.

---

## Page Inventory

| File | Active Nav | Layout Notes |
|---|---|---|
| `index.html` | Video Frames | 1/3 + 2/3 grid |
| `image-enhanced.html` | Image Enhanced | 7/12 + 5/12 grid, sticky right panel |
| `video-enhanced.html` | Video Enhanced | Single column, max-w-900 |
| `image-species-enhanced.html` | Image Species (Enhanced) | 4/12 (upload + enhancement) + 8/12 (detection feed + download) |
| `image-species-direct.html` | Image Species (Direct) | 8/12 + 4/12 upload grid, full-width results |
| `video-species.html` | Video Species | px-8, stepper bar, 4/12 (upload + pipeline config) + 8/12 (frame view + nav + detections + download) |
