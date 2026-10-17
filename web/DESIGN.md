# Tenure demo: design notes

The page should feel like a **retention clerk's printout**: plain, ruled and numerical, and made for someone deciding who to call.

## Tokens (defined at the top of `template.html`)

- **Neutrals:** grey paper (`#f1f1ee`) and near-black ink, not cream. Dark mode uses the same scale inverted (`#141516` paper).
- **Corners:** 0px everywhere. Nothing on the page is rounded.
- **Type:**
  - IBM Plex Sans Condensed for headings.
  - IBM Plex Sans for body text.
  - IBM Plex Mono for every figure.

  Plex was picked because the data is IBM's.
- **One accent:** safety rust (`#b8400c` light, `#f0773f` dark). It means risk and nothing else: the score when it's above the line, effects that push risk up, the flagged squares and the final row of the table. Effects that lower risk stay grey.
- **One signature:** dot leaders between a label and its figure, the way a phone bill prints them. They're used in the summary and on the result slip. Don't add a second signature.

## Things to avoid

- Card shadows.
- Uppercase "eyebrow" labels.
- Pill buttons.
- Coloured status dots.
- Coloured left-border strips.
- A second accent colour.
- Three-stat hero rows.

## Voice

Plain sentences that say what the number means for the person making the call ("of the customers you call were really leaving"). Name the metric in brackets after the plain version. No hype and no em dashes.
