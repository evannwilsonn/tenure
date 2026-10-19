# Tenure demo: design notes

The page is **Swiss data art**, the same piece as the project thumbnail (`.github/thumbnail.jpg`). The motif is a **dot population**: one dot per held-out test customer, coloured by what really happened and ringed when the model flagged them. The flag-line slider redraws it live.

## Tokens (defined at the top of `template.html`)

- **Ground:** warm grey `#e9e5dc` with near-black ink `#1c1b19` (light). The dark theme is a charcoal ground `#1a1917` with warm off-white ink and rings `#ece7dc`.
- **Dots mean outcome:**
  - Orange `#e3570e`, filled: left the company.
  - Thin ink ring: flagged by the model.
  - Small muted dot (`#b7b0a2` light, `#5c574e` dark): stayed.
- **Orange type** (big figures only) is a shade darker in light mode (`#cc4a08`) and a shade lighter in dark mode (`#f06a24`) so it clears 3:1 as large text.
- **Type** (self-hosted from `docs/fonts`, SIL OFL):
  - Bricolage Grotesque Bold for the wordmark, section titles and the big numbers.
  - Bricolage Grotesque Regular for reading.
  - DM Mono for labels, legends, axes and every figure in a table.
- **Controls use the same dot language:**
  - A choice key or switch shows a small grey dot; when chosen it becomes an orange dot with an ink ring.
  - Sliders are a hairline track with an orange, ringed thumb.
- **Churn risk** is a huge orange percentage over a ruler of 50 dots (one per 2%). Dots up to the risk fill orange. The dot at the flag line is ringed.
- **Sections** open with a 2px ink rule, a mono number and a bold title. No cards, shadows or gradients.

## Things to avoid

- Shadows, gradients and glows.
- Orange for small text: it fails contrast on the grey ground. Use it for dots and big figures only.
- Any real telecom company's branding.

## Voice

Plain and exact: "Who is about to cancel?" Results say exactly what was measured.
