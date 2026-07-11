import React, { useEffect, useState } from 'react';
import { FaMapMarkerAlt, FaMagic } from 'react-icons/fa';
import './All.css';

const API_URL = process.env.REACT_APP_API_URL ?? 'http://localhost:8000';
const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const labels = {
  state_class: 'Property class', school_dist: 'School district code', Neighborhood_Code: 'Neighborhood code', Neighborhood_Grp: 'Neighborhood group', Market_Area_1: 'Primary market area', Market_Area_2: 'Secondary market area', econ_area: 'Economic area', econ_bld_class: 'Economic building class', property_use_cd: 'Property use code', impr_tp: 'Improvement type', impr_mdl_cd: 'Building style code', structure: 'Structure class', qa_cd: 'Quality code', yr_impr: 'Improvement year', bld_ar: 'Building area (sq ft)', land_ar: 'Land area', acreage: 'Acreage', date_erected: 'Year built', eff: 'Effective year', yr_remodel: 'Year remodeled', im_sq_ft: 'Improvement area (sq ft)', act_ar: 'Actual area (sq ft)', heat_ar: 'Heated area (sq ft)', gross_ar: 'Gross area (sq ft)', base_ar: 'Base area (sq ft)', accrued_depr_pct: 'Accrued depreciation',
};

export default function HarrisCounty() {
  const [schema, setSchema] = useState(null);
  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/harris-schema/`).then((response) => response.ok ? response.json() : Promise.reject()).then((data) => { setSchema(data); setForm(data.example_input); }).catch(() => setError('Unable to load the Harris County model. Start the Django server and try again.'));
  }, []);
  const change = ({ target }) => setForm((current) => ({ ...current, [target.name]: target.value }));
  const useExample = () => { setForm(schema.example_input); setResult(null); setError(''); };
  const submit = async (event) => {
    event.preventDefault(); setLoading(true); setResult(null); setError('');
    try { const response = await fetch(`${API_URL}/predict-harris/`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) }); const data = await response.json(); if (!response.ok) throw new Error(data.error ?? 'Unable to calculate an estimate.'); setResult(data); } catch (requestError) { setError(requestError.message); } finally { setLoading(false); }
  };
  if (!schema) return <main className="section-shell estimator-page"><p>{error || 'Loading Harris County model…'}</p></main>;
  return <main className="section-shell estimator-page"><div className="estimator-intro"><p className="eyebrow">Texas market model</p><h1>Harris County property estimate.</h1><p>Estimate a 2026 HCAD market value from Harris County property characteristics. This is not a verified sale-price estimate.</p></div><div className="estimator-layout"><section className="estimator-card"><div className="form-toolbar"><div><strong>Harris County profile</strong><span>{schema.features.length} HCAD model fields</span></div><button type="button" className="button button-quiet" onClick={useExample}><FaMagic /> Use example</button></div><form className="estimate-form harris-form" onSubmit={submit}><fieldset><legend><FaMapMarkerAlt /> Property & location codes</legend><div className="field-grid">{schema.features.map((field) => <label className="field" key={field}><span>{labels[field] ?? field}</span><input type={schema.numeric_features.includes(field) ? 'number' : 'text'} step="any" name={field} value={form[field] ?? ''} onChange={change} required={schema.numeric_features.includes(field)} /></label>)}</div></fieldset><button className="button button-primary submit-button" disabled={loading}>{loading ? 'Calculating…' : 'Calculate Harris County estimate'}</button></form></section><aside className="result-column"><section className={`result-card ${result ? 'has-result' : ''}`}>{result ? <><p className="eyebrow">2026 HCAD market value</p><strong className="price">{money.format(result.prediction)}</strong><div className="estimate-range"><span>Likely range</span><strong>{money.format(result.range.low)} – {money.format(result.range.high)}</strong></div><div className="confidence"><span>Confidence</span><strong>{result.confidence.level}</strong></div><p>{result.confidence.message}</p><small className="estimate-disclaimer">{result.disclaimer}</small></> : <><p className="eyebrow">Your result</p><h2>Ready for a Harris County estimate.</h2><p>Use the example profile or enter HCAD property codes and physical characteristics.</p></>}{error && <p className="error-message">{error}</p>}</section><section className="recent-card"><h2>Model evaluation</h2><div className="recent-item"><span>2026 held-out R²</span><strong>{schema.metrics.r2}</strong></div><div className="recent-item"><span>Median percentage error</span><strong>{schema.metrics.median_absolute_percentage_error}%</strong></div></section></aside></div></main>;
}
