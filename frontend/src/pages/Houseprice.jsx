import { useMemo, useState } from 'react';
import { FaArrowRight, FaCheck, FaHome, FaMagic, FaMapMarkerAlt, FaRedo } from 'react-icons/fa';
import { coverageMessage, DEFAULT_MARKET, marketForZip, requestEstimate } from '../model';
import { ALL_FIELD_NAMES, MARKET_FORMS } from './marketForms';
import WhatIf from './WhatIf';
import './All.css';

const formatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const percent = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const signedPercent = (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(1)}%`;
const emptyForm = Object.fromEntries(ALL_FIELD_NAMES.map((name) => [name, '']));
const comparableNotes = {
  'king-county': 'Nearest privacy-safe King County Assessor records—not verified active listings or appraisals.',
  'wake-county': 'Nearest Wake County qualified sales (arm’s-length sales reviewed by county appraisers)—not active listings or appraisals.',
};

function loadRecentEstimates() {
  try {
    return JSON.parse(localStorage.getItem('havenvalue-recent') ?? '[]');
  } catch {
    return [];
  }
}

function saveRecentEstimates(items) {
  try {
    localStorage.setItem('havenvalue-recent', JSON.stringify(items));
  } catch {
    // Storage can be unavailable (private mode); recent estimates are a convenience only.
  }
}

function MarketContext({ context }) {
  if (!context) return null;
  const month = context.month ? new Date(`${context.month}-15T00:00:00Z`).toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' }) : '';
  return (
    <section className="recent-card" aria-labelledby="market-title">
      <h2 id="market-title"><FaMapMarkerAlt /> Market in ZIP {context.zip}</h2>
      <p className="factor-note">Latest monthly figures{month ? ` (${month})` : ''}. The estimate scales with this ZIP’s Zillow home value index.</p>
      {context.zhvi != null && <div className="recent-item"><span>Typical home value (Zillow)</span><strong>{formatter.format(context.zhvi)}{context.zhvi_yoy != null && <small> {signedPercent(context.zhvi_yoy)} / yr</small>}</strong></div>}
      {context.days_on_market != null && <div className="recent-item"><span>Median days on market</span><strong>{Math.round(context.days_on_market)}</strong></div>}
      {context.price_reduced_share != null && <div className="recent-item"><span>Listings with a price cut</span><strong>{percent(context.price_reduced_share)}</strong></div>}
      {context.inventory_yoy != null && <div className="recent-item"><span>Active listings vs. a year ago</span><strong>{signedPercent(context.inventory_yoy)}</strong></div>}
      {context.mortgage_rate_30y != null && <div className="recent-item"><span>30-year fixed rate (Freddie Mac)</span><strong>{context.mortgage_rate_30y.toFixed(2)}%</strong></div>}
    </section>
  );
}

function Houseprice() {
  const [formData, setFormData] = useState(emptyForm);
  const [result, setResult] = useState(null);
  const [estimatedInputs, setEstimatedInputs] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [recent, setRecent] = useState(loadRecentEstimates);

  const zipText = String(formData.zip_code ?? '').trim();
  const detectedMarket = marketForZip(zipText);
  const marketId = detectedMarket?.id ?? DEFAULT_MARKET;
  const form = MARKET_FORMS[marketId];
  const uncovered = /^\d{5}$/.test(zipText) && !detectedMarket;
  const completed = useMemo(() => form.fields.filter(({ name }) => formData[name]).length, [form, formData]);

  const clearResult = () => { setResult(null); setEstimatedInputs(null); setError(''); };
  const handleChange = ({ target: { name, value } }) => setFormData((current) => ({ ...current, [name]: value }));
  const useSample = () => { setFormData({ ...emptyForm, ...form.sample }); clearResult(); };
  const reset = () => { setFormData(emptyForm); clearResult(); };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true); clearResult();
    // Only this market's fields are sent; the model runs in the browser (src/model).
    const payload = Object.fromEntries(form.fields.map(({ name }) => [name, formData[name]]));
    try {
      const estimate = await requestEstimate(payload);
      setResult(estimate); setEstimatedInputs(payload);
      const nextRecent = [{ value: estimate.prediction, profile: `${MARKET_FORMS[estimate.market].profile(payload)} · ${estimate.market_name}`, date: new Date().toLocaleDateString() }, ...recent].slice(0, 3);
      setRecent(nextRecent); saveRecentEstimates(nextRecent);
    } catch (estimateError) { setError(estimateError.message || 'Unable to calculate an estimate right now.'); } finally { setLoading(false); }
  };

  const groups = [...new Set(form.fields.map(({ group }) => group))];
  const prediction = result?.prediction ?? null;
  const estimateRange = result?.range ?? null;
  const confidence = result?.confidence ?? null;
  const modelFactors = result?.top_factors ?? result?.model_factors ?? [];
  const comparables = result?.comparables ?? [];
  const resultForm = result ? MARKET_FORMS[result.market] : form;
  return (
    <main className="estimator-page section-shell">
      <div className="estimator-intro"><p className="eyebrow">Free home estimate</p><h1>Start with the details you know.</h1><p>Enter a ZIP code in Seattle or Wake County, NC, then complete the property profile. Your progress is saved only in this browser.</p></div>
      <div className="estimator-layout">
        <section className="estimator-card">
          <div className="form-toolbar"><div><strong>Property profile</strong><span>{completed} of {form.fields.length} fields complete</span></div><div className="toolbar-actions"><button type="button" className="button button-quiet" onClick={useSample}><FaMagic /> Use sample</button><button type="button" className="icon-button" onClick={reset} aria-label="Reset form"><FaRedo /></button></div></div>
          <div className="progress-track"><span style={{ width: `${(completed / form.fields.length) * 100}%` }} /></div>
          {detectedMarket && <p className="market-note" data-testid="market-note"><FaMapMarkerAlt /> Using the {detectedMarket.name} model</p>}
          {uncovered && <p className="market-note market-note-warning" role="status">{coverageMessage()}</p>}
          <form className="estimate-form" onSubmit={handleSubmit}>
            {groups.map((group) => <fieldset key={group}><legend>{group}</legend><div className="field-grid">{form.fields.filter((field) => field.group === group).map((field) => <label className="field" key={field.name}><span>{field.label}{field.required === false && <em> optional</em>}</span>{field.kind === 'select' ? <select name={field.name} value={formData[field.name]} onChange={handleChange} required={field.required !== false}><option value="">Select one</option>{field.options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select> : <input type="number" name={field.name} value={formData[field.name]} onChange={handleChange} min={field.min} max={field.max} step={field.step} required={field.required !== false} />}{field.hint && <small>{field.hint}</small>}</label>)}</div></fieldset>)}
            <button className="button button-primary submit-button" type="submit" disabled={loading}>{loading ? 'Calculating estimate…' : <>Calculate estimate <FaArrowRight /></>}</button>
          </form>
        </section>
        <aside className="result-column">
          <section className={`result-card ${prediction !== null ? 'has-result' : ''}`} aria-live="polite">{prediction === null ? <><span className="result-icon"><FaHome /></span><p className="eyebrow">Your result</p><h2>Your estimate will appear here.</h2><p>Enter your property details and we’ll calculate a data-informed value.</p></> : <><span className="result-icon"><FaCheck /></span><p className="eyebrow">Estimated value</p><strong className="price">{formatter.format(prediction)}</strong>{estimateRange && <div className="estimate-range"><span>{estimateRange.nominal_coverage ? `${Math.round(estimateRange.nominal_coverage * 100)}% empirical range` : 'Likely range'}</span><strong>{formatter.format(estimateRange.low)} – {formatter.format(estimateRange.high)}</strong></div>}{confidence && <div className="confidence"><span>Input confidence</span><strong>{confidence.level}</strong></div>}<p>{confidence?.message}</p><small className="estimate-disclaimer">{result.disclaimer}</small></>}{error && <p className="error-message" role="alert">{error}</p>}</section>
          {prediction !== null && estimatedInputs && <WhatIf market={result.market} inputs={estimatedInputs} basePrediction={prediction} />}
          {result?.market_context && <MarketContext context={result.market_context} />}
          {modelFactors.length > 0 && <section className="recent-card"><h2>What shaped this estimate</h2><p className="factor-note">Per-property contributions compared with a typical {resultForm.placeLabel} property. Positive values raise the estimate; negative values lower it.</p>{modelFactors.slice(0, 8).map((factor) => <div className="recent-item" key={factor.feature}><span>{factor.label ?? factor.feature.replaceAll('_', ' ')}</span><strong className={(factor.contribution ?? 0) >= 0 ? 'positive-factor' : 'negative-factor'}>{factor.contribution !== undefined ? `${factor.contribution >= 0 ? '+' : '−'}${formatter.format(Math.abs(factor.contribution))}` : formatter.format(factor.abs_importance ?? Math.abs(factor.coefficient ?? 0))}</strong></div>)}</section>}
          {comparables.length > 0 && <section className="recent-card"><h2>Similar historical sales</h2><p className="factor-note">{comparableNotes[result.market]}</p>{comparables.map((property, index) => <div className="comparable-item" key={`${property.zip_code}-${property.price}-${index}`}><div><strong>{formatter.format(property.price)}</strong><span>{property.beds !== undefined && `${property.beds} bd · `}{property.baths} ba · {Number(property.size).toLocaleString()} sq ft{property.property_type ? ` · ${property.property_type}` : ''}</span></div><div><span>ZIP {property.zip_code}</span><small>{Math.round(property.similarity * 100)}% similar</small></div></div>)}</section>}
          {recent.length > 0 && <section className="recent-card"><h2>Recent estimates</h2>{recent.map((item, index) => <div className="recent-item" key={`${item.date}-${index}`}><span>{item.profile ?? 'Property profile'} · {item.date}</span><strong>{formatter.format(item.value)}</strong></div>)}</section>}
        </aside>
      </div>
    </main>
  );
}

export default Houseprice;
