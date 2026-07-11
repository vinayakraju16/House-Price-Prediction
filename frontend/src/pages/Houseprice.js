import React, { useEffect, useMemo, useState } from 'react';
import { FaArrowRight, FaCheck, FaHome, FaMapMarkerAlt, FaMagic, FaRedo } from 'react-icons/fa';
import './All.css';

const API_URL = process.env.REACT_APP_API_URL ?? 'http://localhost:8000';
const formatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const fields = [
  { name: 'OverallQual', label: 'Overall quality', hint: '1 = basic, 10 = exceptional', min: 1, max: 10, group: 'Property basics' },
  { name: 'OverallCond', label: 'Overall condition', hint: '1 = poor, 10 = excellent', min: 1, max: 10, group: 'Property basics' },
  { name: 'YearBuilt', label: 'Year built', hint: 'Four-digit year', min: 1800, max: new Date().getFullYear(), group: 'Property basics' },
  { name: 'LotArea', label: 'Lot area', hint: 'Square feet', min: 0, group: 'Property basics' },
  { name: 'TotRmsAbvGrd', label: 'Rooms above ground', hint: 'Exclude bathrooms', min: 1, group: 'Living space' },
  { name: 'MasVnrArea', label: 'Masonry veneer area', hint: 'Square feet', min: 0, group: 'Living space' },
  { name: 'BsmtFinSF1', label: 'Finished basement area', hint: 'Square feet', min: 0, group: 'Living space' },
  { name: 'BsmtUnfSF', label: 'Unfinished basement area', hint: 'Square feet', min: 0, group: 'Living space' },
  { name: 'GarageCars', label: 'Garage capacity', hint: 'Number of cars', min: 0, max: 5, group: 'Garage & features' },
  { name: 'GarageArea', label: 'Garage area', hint: 'Square feet', min: 0, group: 'Garage & features' },
  { name: 'SaleType_WD', label: 'Sale type', kind: 'select', options: [['1', 'Warranty deed'], ['0', 'Other']], group: 'Garage & features' },
  { name: 'ExterQual_Gd', label: 'Exterior quality', kind: 'select', options: [['1', 'Good'], ['0', 'Other']], group: 'Garage & features' },
  { name: 'ExterCond_Fa', label: 'Exterior condition', kind: 'select', options: [['1', 'Fair'], ['0', 'Other']], group: 'Garage & features' },
  { name: 'BsmtFinType1_LwQ', label: 'Basement finish', kind: 'select', options: [['1', 'Low quality'], ['0', 'Other']], group: 'Garage & features' },
];

const sampleHome = { MasVnrArea: '196', SaleType_WD: '1', OverallQual: '7', OverallCond: '5', ExterQual_Gd: '1', ExterCond_Fa: '0', BsmtUnfSF: '150', BsmtFinType1_LwQ: '0', LotArea: '8450', YearBuilt: '2003', BsmtFinSF1: '706', TotRmsAbvGrd: '8', GarageCars: '2', GarageArea: '548' };
const emptyForm = Object.fromEntries(fields.map(({ name }) => [name, '']));

function Houseprice() {
  const [formData, setFormData] = useState(emptyForm);
  const [prediction, setPrediction] = useState(null);
  const [estimateRange, setEstimateRange] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [modelFactors, setModelFactors] = useState([]);
  const [disclaimer, setDisclaimer] = useState('');
  const [address, setAddress] = useState('');
  const [market, setMarket] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [recent, setRecent] = useState([]);
  const completed = useMemo(() => Object.values(formData).filter(Boolean).length, [formData]);

  useEffect(() => {
    try { setRecent(JSON.parse(localStorage.getItem('havenvalue-recent') ?? '[]')); } catch { setRecent([]); }
  }, []);

  const handleChange = ({ target: { name, value } }) => setFormData((current) => ({ ...current, [name]: value }));
  const useSample = () => { setFormData(sampleHome); setAddress('8450 Sunset Trail, Ames, IA'); setPrediction(null); setEstimateRange(null); setConfidence(null); setModelFactors([]); setDisclaimer(''); setMarket(null); setError(''); };
  const reset = () => { setFormData(emptyForm); setAddress(''); setPrediction(null); setEstimateRange(null); setConfidence(null); setModelFactors([]); setDisclaimer(''); setMarket(null); setError(''); };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true); setError(''); setPrediction(null); setEstimateRange(null); setConfidence(null); setModelFactors([]); setDisclaimer(''); setMarket(null);
    try {
      const response = await fetch(`${API_URL}/predict/`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...formData, address }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? 'Unable to get an estimate right now.');
      setPrediction(result.prediction); setEstimateRange(result.range ?? null); setConfidence(result.confidence ?? null); setModelFactors(result.model_factors ?? []); setDisclaimer(result.disclaimer ?? ''); setMarket(result.market ?? null);
      const nextRecent = [{ value: result.prediction, quality: formData.OverallQual, date: new Date().toLocaleDateString() }, ...recent].slice(0, 3);
      setRecent(nextRecent); localStorage.setItem('havenvalue-recent', JSON.stringify(nextRecent));
    } catch (requestError) { setError(requestError.message); } finally { setLoading(false); }
  };

  const groups = [...new Set(fields.map(({ group }) => group))];
  return (
    <main className="estimator-page section-shell">
      <div className="estimator-intro"><p className="eyebrow">Free home estimate</p><h1>Start with the details you know.</h1><p>Complete the property profile for a personalized estimate. Your progress is saved only in this browser.</p></div>
      <div className="estimator-layout">
        <section className="estimator-card">
          <div className="form-toolbar"><div><strong>Property profile</strong><span>{completed} of {fields.length} fields complete</span></div><div className="toolbar-actions"><button type="button" className="button button-quiet" onClick={useSample}><FaMagic /> Use sample</button><button type="button" className="icon-button" onClick={reset} aria-label="Reset form"><FaRedo /></button></div></div>
          <div className="progress-track"><span style={{ width: `${(completed / fields.length) * 100}%` }} /></div>
          <form className="estimate-form" onSubmit={handleSubmit}>
            <div className="address-field"><label htmlFor="address"><FaMapMarkerAlt /> Property address <em>optional</em></label><input id="address" type="text" value={address} onChange={(event) => setAddress(event.target.value)} placeholder="Street, city, state, ZIP" /><small>Add a U.S. address to see live county context. It does not change your model estimate yet.</small></div>
            {groups.map((group) => <fieldset key={group}><legend>{group}</legend><div className="field-grid">{fields.filter((field) => field.group === group).map((field) => <label className="field" key={field.name}><span>{field.label}</span>{field.kind === 'select' ? <select name={field.name} value={formData[field.name]} onChange={handleChange} required><option value="">Select one</option>{field.options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select> : <input type="number" name={field.name} value={formData[field.name]} onChange={handleChange} min={field.min} max={field.max} required />}{field.hint && <small>{field.hint}</small>}</label>)}</div></fieldset>)}
            <button className="button button-primary submit-button" type="submit" disabled={loading}>{loading ? 'Calculating estimate…' : <>Calculate estimate <FaArrowRight /></>}</button>
          </form>
        </section>
        <aside className="result-column">
          <section className={`result-card ${prediction !== null ? 'has-result' : ''}`} aria-live="polite">{prediction === null ? <><span className="result-icon"><FaHome /></span><p className="eyebrow">Your result</p><h2>Your estimate will appear here.</h2><p>Enter your property details and we’ll calculate a data-informed value.</p></> : <><span className="result-icon"><FaCheck /></span><p className="eyebrow">Estimated value</p><strong className="price">{formatter.format(prediction)}</strong>{estimateRange && <div className="estimate-range"><span>Likely range</span><strong>{formatter.format(estimateRange.low)} – {formatter.format(estimateRange.high)}</strong></div>}{confidence && <div className="confidence"><span>Confidence</span><strong>{confidence.level}</strong></div>}<p>{confidence?.message}</p><small className="estimate-disclaimer">{disclaimer}</small></>}{error && <p className="error-message" role="alert">{error}</p>}</section>
          {market && <section className="market-card"><p className="eyebrow">Live market context</p><h2>{market.available ? market.county : 'Lookup unavailable'}</h2>{market.available ? <><p>{market.address}</p>{market.county_median_home_value && <div className="market-stat"><span>County median home value</span><strong>{formatter.format(market.county_median_home_value)}</strong></div>}{market.mortgage_rate && <div className="market-stat"><span>Latest 30-year mortgage rate</span><strong>{market.mortgage_rate}%</strong></div>}<small>{market.market_data_note ?? market.updated_at}</small></> : <p>{market.message}</p>}</section>}
          {modelFactors.length > 0 && <section className="recent-card"><h2>Top model factors</h2><p className="factor-note">These are globally important features in the training data, not individual price adjustments.</p>{modelFactors.map((factor) => <div className="recent-item" key={factor.feature}><span>{factor.feature.replace(/([A-Z])/g, ' $1').trim()}</span><strong>{Math.round(factor.importance * 100)}%</strong></div>)}</section>}
          {recent.length > 0 && <section className="recent-card"><h2>Recent estimates</h2>{recent.map((item, index) => <div className="recent-item" key={`${item.date}-${index}`}><span>Quality {item.quality} · {item.date}</span><strong>{formatter.format(item.value)}</strong></div>)}</section>}
        </aside>
      </div>
    </main>
  );
}

export default Houseprice;
