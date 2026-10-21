import { useEffect, useMemo, useState } from 'react';
import { FaRedo, FaSlidersH } from 'react-icons/fa';
import { loadModel } from '../model';
import { predictPayload } from '../model/predictor';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const grades = { 6: 'Low', 7: 'Average', 8: 'Good', 9: 'Better', 10: 'Very good', 11: 'Excellent', 12: 'Luxury', 13: 'Exceptional' };
const conditions = { 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Very good' };
const currentYear = new Date().getFullYear();

const sliders = [
  { name: 'size', label: 'Living area', min: 400, max: 8000, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
  { name: 'grade', label: 'Building grade', min: 6, max: 13, step: 1, format: (v) => `${v} — ${grades[v]}` },
  { name: 'condition', label: 'Condition', min: 1, max: 5, step: 1, format: (v) => conditions[v] },
  { name: 'finished_basement_sqft', label: 'Finished basement', min: 0, max: 2500, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
  { name: 'garage_sqft', label: 'Garage area', min: 0, max: 1200, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
];

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

/** Starting values: what the user entered, else the training default the model used. */
function startingValues(inputs, defaults) {
  const value = (name) => (inputs[name] !== '' && inputs[name] !== undefined ? Number(inputs[name]) : Number(defaults[name]));
  const start = {};
  // Not clamped: the sliders widen to fit unusual inputs so the first adjusted estimate matches.
  for (const slider of sliders) start[slider.name] = value(slider.name);
  start.has_view = value('has_view') === 1;
  const builtYear = clamp(value('year_built'), 1800, currentYear);
  start.renovated = inputs.year_renovated !== '' && inputs.year_renovated !== undefined && Number(inputs.year_renovated) !== 0;
  start.year_renovated = start.renovated ? clamp(Number(inputs.year_renovated), builtYear, currentYear) : currentYear;
  start.built_year = builtYear;
  return start;
}

/**
 * Re-prices the home instantly as the user changes key features. Every change re-runs the
 * full gradient-boosted model in the browser (well under a millisecond per estimate).
 */
export default function WhatIf({ inputs, basePrediction }) {
  const [model, setModel] = useState(null);
  const [values, setValues] = useState(null);

  useEffect(() => {
    let active = true;
    loadModel().then((loaded) => {
      if (!active) return;
      setModel(loaded);
      setValues(startingValues(inputs, loaded.feature_defaults));
    }).catch(() => {});
    return () => { active = false; };
  }, [inputs]);

  const adjusted = useMemo(() => {
    if (!model || !values) return null;
    const payload = { ...inputs, has_view: values.has_view ? 1 : 0, year_renovated: values.renovated ? values.year_renovated : '' };
    for (const slider of sliders) payload[slider.name] = values[slider.name];
    try {
      return Math.round(predictPayload(model, payload) * 100) / 100;
    } catch {
      return null;
    }
  }, [model, values, inputs]);

  if (!model || !values) return null;
  const change = (name, value) => setValues((current) => ({ ...current, [name]: value }));
  const delta = adjusted === null ? 0 : adjusted - basePrediction;
  const percent = basePrediction ? (delta / basePrediction) * 100 : 0;
  const deltaText = Math.abs(delta) < 0.5 ? 'Same as your estimate' : `${delta > 0 ? '+' : '−'}${money.format(Math.abs(delta))} (${delta > 0 ? '+' : '−'}${Math.abs(percent).toFixed(1)}%)`;

  return (
    <section className="recent-card whatif-card" aria-labelledby="whatif-title">
      <div className="whatif-header">
        <h2 id="whatif-title"><FaSlidersH /> What if…?</h2>
        <button type="button" className="icon-button" onClick={() => setValues(startingValues(inputs, model.feature_defaults))} aria-label="Reset what-if changes"><FaRedo /></button>
      </div>
      <p className="factor-note">Change a feature and the model re-prices this home instantly, right in your browser.</p>
      <div className="whatif-result" aria-live="polite">
        <span>Adjusted estimate</span>
        <strong data-testid="whatif-estimate">{adjusted === null ? '—' : money.format(adjusted)}</strong>
        <small className={delta >= 0 ? 'positive-factor' : 'negative-factor'}>{deltaText}</small>
      </div>
      {sliders.map((slider) => (
        <label className="whatif-slider" key={slider.name}>
          <span><b>{slider.label}</b><em>{slider.format(values[slider.name])}</em></span>
          <input
            type="range"
            min={Math.min(slider.min, values[slider.name])}
            max={Math.max(slider.max, values[slider.name])}
            step={slider.step}
            value={values[slider.name]}
            aria-valuetext={String(slider.format(values[slider.name]))}
            onChange={(event) => change(slider.name, Number(event.target.value))}
          />
        </label>
      ))}
      <label className="whatif-toggle">
        <input type="checkbox" checked={values.has_view} onChange={(event) => change('has_view', event.target.checked)} />
        <span>Scenic view</span>
      </label>
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
