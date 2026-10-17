# Tenure demo: design notes

The page is a **vintage telecom test console**. You set up a customer with hardware controls and read the churn risk off an analog meter. Lit lamps explain the reading.

## Tokens (defined at the top of `template.html`)

- **Panels:** warm beige instrument panels (`#d6d1c4`) on a darker bench, with bevelled edges and four screws each. The night panel is black anodised (`#1d1e21`).
- **Controls:**
  - Faders for numbers.
  - Push keys with an indicator dot for choices.
  - Lever switches for yes/no.
  - Black memory buttons for presets.
- **Type:**
  - Barlow Condensed, uppercase and tracked, for engraved labels.
  - Barlow for reading.
  - Share Tech Mono for every digital readout.
- **Colour means state:**
  - Amber: the lit key or lever, readouts, and inputs pushing risk up.
  - Green: inputs holding risk down.
  - Red: the flag lamp and the meter's zone past the flag line.

  The glow on lamps and LEDs is how a lamp shows "on", so it appears only on lit indicators.
- **The meter** is the centrepiece: a cream dial, a red needle and a red zone that moves with the flag line.

## Things to avoid

- Glow on anything that isn't a lit indicator.
- Gradients beyond the panel bevel.
- Any real telecom company's branding.
- Skeuomorphic textures beyond the brushed bevel and screws.

## Voice

Bench-instrument plain: "Set up a customer. Watch the needle." Results say exactly what was measured.
