import { useEffect, useMemo, useState } from 'react';
import { FaRedo, FaSlidersH } from 'react-icons/fa';
import { DEFAULT_MARKET, loadModel, predictWithModel } from '../model';
import { MARKET_FORMS } from './marketForms';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const currentYear = new Date().getFullYear();
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

/** Slider position for a value; discrete sliders (`steps`) move over list positions. */
const toPosition = (slider, value) => (slider.steps
  ? slider.steps.reduce((best, step, index) => (Math.abs(step - value) < Math.abs(slider.steps[best] - value) ? index : best), 0)
  : value);
const fromPosition = (slider, position) => (slider.steps ? slider.steps[position] : position);

/** Starting values: what the user entered, else the training default the model used. */
function startingValues(controls, inputs, defaults) {
  const value = (name) => (inputs[name] !== '' && inputs[name] !== undefined ? Number(inputs[name]) : Number(defaults[name]));
  const start = {};
  // Not clamped: the sliders widen to fit unusual inputs so the first adjusted estimate matches.
  for (const slider of controls.sliders) start[slider.name] = value(slider.name);
  for (const toggle of controls.toggles) start[toggle.name] = value(toggle.name) === 1;
  const builtYear = clamp(value('year_built'), 1800, currentYear);
  start.renovated = inputs.year_renovated !== '' && inputs.year_renovated !== undefined && Number(inputs.year_renovated) !== 0;
  start.year_renovated = start.renovated ? clamp(Number(inputs.year_renovated), builtYear, currentYear) : currentYear;
  start.built_year = builtYear;
  return start;
}

/**
 * Re-prices the home instantly as the user changes key features. Every change re-runs the
 * market's full gradient-boosted model in the browser (well under a millisecond each).
 */
export default function WhatIf({ market = DEFAULT_MARKET, inputs, basePrediction }) {
  const controls = MARKET_FORMS[market];
  const [model, setModel] = useState(null);
  const [values, setValues] = useState(null);

  useEffect(() => {
    let active = true;
    loadModel(market).then((loaded) => {
      if (!active) return;
      setModel(loaded);
      setValues(startingValues(controls, inputs, loaded.feature_defaults));
    }).catch(() => {});
    return () => { active = false; };
  }, [market, controls, inputs]);

  const adjusted = useMemo(() => {
    if (!model || !values) return null;
    const payload = { ...inputs, year_renovated: values.renovated ? values.year_renovated : '' };
    for (const slider of controls.sliders) payload[slider.name] = values[slider.name];
    for (const toggle of controls.toggles) payload[toggle.name] = values[toggle.name] ? 1 : 0;
    try {
      return Math.round(predictWithModel(market, model, payload) * 100) / 100;
    } catch {
      return null;
    }
  }, [market, controls, model, values, inputs]);

  if (!model || !values) return null;
  const change = (name, value) => setValues((current) => ({ ...current, [name]: value }));
  const delta = adjusted === null ? 0 : adjusted - basePrediction;
  const percent = basePrediction ? (delta / basePrediction) * 100 : 0;
  const deltaText = Math.abs(delta) < 0.5 ? 'Same as your estimate' : `${delta > 0 ? '+' : '−'}${money.format(Math.abs(delta))} (${delta > 0 ? '+' : '−'}${Math.abs(percent).toFixed(1)}%)`;

  return (
    <section className="recent-card whatif-card" aria-labelledby="whatif-title">
      <div className="whatif-header">
        <h2 id="whatif-title"><FaSlidersH /> What if…?</h2>
        <button type="button" className="icon-button" onClick={() => setValues(startingValues(controls, inputs, model.feature_defaults))} aria-label="Reset what-if changes"><FaRedo /></button>
      </div>
      <p className="factor-note">Change a feature and the model re-prices this home instantly, right in your browser.</p>
      <div className="whatif-result" aria-live="polite">
        <span>Adjusted estimate</span>
        <strong data-testid="whatif-estimate">{adjusted === null ? '—' : money.format(adjusted)}</strong>
        <small className={delta >= 0 ? 'positive-factor' : 'negative-factor'}>{deltaText}</small>
      </div>
      {controls.sliders.map((slider) => {
        const position = toPosition(slider, values[slider.name]);
        const min = slider.steps ? 0 : Math.min(slider.min, values[slider.name]);
        const max = slider.steps ? slider.steps.length - 1 : Math.max(slider.max, values[slider.name]);
        return (
          <label className="whatif-slider" key={slider.name}>
            <span><b>{slider.label}</b><em>{slider.format(values[slider.name])}</em></span>
            <input
              type="range"
              min={min}
              max={max}
              step={slider.steps ? 1 : slider.step}
              value={position}
              aria-valuetext={String(slider.format(values[slider.name]))}
              onChange={(event) => change(slider.name, fromPosition(slider, Number(event.target.value)))}
            />
          </label>
        );
      })}
      {controls.toggles.map((toggle) => (
        <label className="whatif-toggle" key={toggle.name}>
          <input type="checkbox" checked={values[toggle.name]} onChange={(event) => change(toggle.name, event.target.checked)} />
          <span>{toggle.label}</span>
        </label>
      ))}
      <label className="whatif-toggle">
        <input type="checkbox" checked={values.renovated} onChange={(event) => change('renovated', event.target.checked)} />
        <span>Renovated</span>
      </label>
      {values.renovated && (
        <label className="whatif-slider">
          <span><b>Year renovated</b><em>{values.year_renovated}</em></span>
          <input
            type="range"
            min={values.built_year}
            max={currentYear}
            step={1}
            value={values.year_renovated}
            onChange={(event) => change('year_renovated', Number(event.target.value))}
          />
        </label>
      )}
    </section>
  );
}
