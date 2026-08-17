import { useMemo, useState } from 'react';
import { FaArrowRight, FaCheck, FaHome, FaMapMarkerAlt, FaMagic, FaRedo } from 'react-icons/fa';
import { API_URL } from '../config';
import './All.css';

const formatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const fields = [
  { name: 'beds', label: 'Bedrooms', hint: 'Number of bedrooms', min: 1, step: 1, group: 'Property basics' },
  { name: 'baths', label: 'Bathrooms', hint: 'Number of bathrooms', min: 0.5, step: 0.5, group: 'Property basics' },
  { name: 'size', label: 'Home size', hint: 'Square feet', min: 1, step: 1, group: 'Property basics' },
  { name: 'lot_size', label: 'Lot size', hint: 'Square feet; use 0 when unavailable', min: 0, step: 1, group: 'Property basics' },
  { name: 'zip_code', label: 'ZIP code', hint: 'Seattle-area ZIP from 98101 to 98199', min: 98101, max: 98199, step: 1, group: 'Location' },
  { name: 'stories', label: 'Stories', hint: 'Above-ground story count', min: 0.5, max: 10, step: 0.5, group: 'Building details', required: false },
  { name: 'grade', label: 'Building grade', hint: 'Construction quality recorded by the assessor', kind: 'select', group: 'Building details', required: false, options: [['6', '6 — Low'], ['7', '7 — Average'], ['8', '8 — Good'], ['9', '9 — Better'], ['10', '10 — Very good'], ['11', '11 — Excellent'], ['12', '12 — Luxury'], ['13', '13 — Exceptional']] },
  { name: 'condition', label: 'Condition', hint: 'Current physical condition', kind: 'select', group: 'Building details', required: false, options: [['1', 'Poor'], ['2', 'Fair'], ['3', 'Average'], ['4', 'Good'], ['5', 'Very good']] },
  { name: 'year_built', label: 'Year built', min: 1800, max: new Date().getFullYear(), step: 1, group: 'Age and renovation', required: false },
  { name: 'year_renovated', label: 'Year renovated', hint: 'Leave blank if never renovated', min: 1800, max: new Date().getFullYear(), step: 1, group: 'Age and renovation', required: false },
  { name: 'finished_basement_sqft', label: 'Finished basement', hint: 'Finished basement square feet', min: 0, step: 1, group: 'Additional spaces', required: false },
  { name: 'garage_sqft', label: 'Garage area', hint: 'Attached and basement garage square feet', min: 0, step: 1, group: 'Additional spaces', required: false },
  { name: 'fireplaces', label: 'Fireplaces', min: 0, max: 30, step: 1, group: 'Additional spaces', required: false },
  { name: 'has_view', label: 'Scenic view', hint: 'Whether the property has a recorded view', kind: 'select', group: 'Additional spaces', required: false, options: [['0', 'No'], ['1', 'Yes']] },
];

const sampleHome = {
  beds: '3', baths: '2.5', size: '2590', lot_size: '6000', zip_code: '98144',
  stories: '2', grade: '8', condition: '3', year_built: '1996', year_renovated: '',
  finished_basement_sqft: '450', garage_sqft: '400', fireplaces: '1', has_view: '0',
};
const emptyForm = Object.fromEntries(fields.map(({ name }) => [name, '']));

function apiErrorMessage(result) {
  if (result.message) return result.message;
  const firstFieldError = Object.values(result.details ?? {})[0];
  return firstFieldError ?? result.error ?? 'Unable to get an estimate right now.';
}

function loadRecentEstimates() {
  try {
    return JSON.parse(localStorage.getItem('havenvalue-recent') ?? '[]');
  } catch {
    return [];
  }
}

function Houseprice() {
  const [formData, setFormData] = useState(emptyForm);
  const [prediction, setPrediction] = useState(null);
  const [estimateRange, setEstimateRange] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [modelFactors, setModelFactors] = useState([]);
  const [comparables, setComparables] = useState([]);
  const [disclaimer, setDisclaimer] = useState('');
  const [pincode, setPincode] = useState('');
  const [market, setMarket] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [recent, setRecent] = useState(loadRecentEstimates);
  const completed = useMemo(() => Object.values(formData).filter(Boolean).length, [formData]);

  const handleChange = ({ target: { name, value } }) => setFormData((current) => ({ ...current, [name]: value }));
  const useSample = () => { setFormData(sampleHome); setPincode('50010'); setPrediction(null); setEstimateRange(null); setConfidence(null); setModelFactors([]); setComparables([]); setDisclaimer(''); setMarket(null); setError(''); };
  const reset = () => { setFormData(emptyForm); setPincode(''); setPrediction(null); setEstimateRange(null); setConfidence(null); setModelFactors([]); setComparables([]); setDisclaimer(''); setMarket(null); setError(''); };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true); setError(''); setPrediction(null); setEstimateRange(null); setConfidence(null); setModelFactors([]); setComparables([]); setDisclaimer(''); setMarket(null);
    try {
      const response = await fetch(`${API_URL}/predict/`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...formData, pincode }) });
      const result = await response.json();
      if (!response.ok) throw new Error(apiErrorMessage(result));
      setPrediction(result.prediction); setEstimateRange(result.range ?? null); setConfidence(result.confidence ?? null); setModelFactors(result.top_factors ?? result.model_factors ?? []); setComparables(result.comparables ?? []); setDisclaimer(result.disclaimer ?? ''); setMarket(result.market ?? null);
      const nextRecent = [{ value: result.prediction, profile: `${formData.beds} bd / ${formData.baths} ba`, date: new Date().toLocaleDateString() }, ...recent].slice(0, 3);
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
            <div className="address-field"><label htmlFor="pincode"><FaMapMarkerAlt /> ZIP / pincode <em>optional</em></label><input id="pincode" type="text" value={pincode} onChange={(event) => setPincode(event.target.value)} placeholder="e.g. 77002 or 50010" /><small>Enter a ZIP or postal code to associate the estimate with a location. The estimate itself is still based on your property profile.</small></div>
            {groups.map((group) => <fieldset key={group}><legend>{group}</legend><div className="field-grid">{fields.filter((field) => field.group === group).map((field) => <label className="field" key={field.name}><span>{field.label}{field.required === false && <em> optional</em>}</span>{field.kind === 'select' ? <select name={field.name} value={formData[field.name]} onChange={handleChange} required={field.required !== false}><option value="">Select one</option>{field.options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select> : <input type="number" name={field.name} value={formData[field.name]} onChange={handleChange} min={field.min} max={field.max} step={field.step} required={field.required !== false} />}{field.hint && <small>{field.hint}</small>}</label>)}</div></fieldset>)}
            <button className="button button-primary submit-button" type="submit" disabled={loading}>{loading ? 'Calculating estimate…' : <>Calculate estimate <FaArrowRight /></>}</button>
          </form>
        </section>
        <aside className="result-column">
          <section className={`result-card ${prediction !== null ? 'has-result' : ''}`} aria-live="polite">{prediction === null ? <><span className="result-icon"><FaHome /></span><p className="eyebrow">Your result</p><h2>Your estimate will appear here.</h2><p>Enter your property details and we’ll calculate a data-informed value.</p></> : <><span className="result-icon"><FaCheck /></span><p className="eyebrow">Estimated value</p><strong className="price">{formatter.format(prediction)}</strong>{estimateRange && <div className="estimate-range"><span>{estimateRange.nominal_coverage ? `${Math.round(estimateRange.nominal_coverage * 100)}% empirical range` : 'Likely range'}</span><strong>{formatter.format(estimateRange.low)} – {formatter.format(estimateRange.high)}</strong></div>}{confidence && <div className="confidence"><span>Input confidence</span><strong>{confidence.level}</strong></div>}<p>{confidence?.message}</p><small className="estimate-disclaimer">{disclaimer}</small></>}{error && <p className="error-message" role="alert">{error}</p>}</section>
          {market && <section className="market-card"><p className="eyebrow">Live market context</p><h2>{market.available ? market.county : 'Lookup unavailable'}</h2>{market.available ? <><p>{market.address}</p>{market.county_median_home_value && <div className="market-stat"><span>County median home value</span><strong>{formatter.format(market.county_median_home_value)}</strong></div>}{market.mortgage_rate && <div className="market-stat"><span>Latest 30-year mortgage rate</span><strong>{market.mortgage_rate}%</strong></div>}<small>{market.market_data_note ?? market.updated_at}</small></> : <p>{market.message}</p>}</section>}
          {modelFactors.length > 0 && <section className="recent-card"><h2>What shaped this estimate</h2><p className="factor-note">Per-property contributions compared with a typical Seattle property. Positive values raise the estimate; negative values lower it.</p>{modelFactors.slice(0, 8).map((factor) => <div className="recent-item" key={factor.feature}><span>{factor.label ?? factor.feature.replaceAll('_', ' ')}</span><strong className={(factor.contribution ?? 0) >= 0 ? 'positive-factor' : 'negative-factor'}>{factor.contribution !== undefined ? `${factor.contribution >= 0 ? '+' : '−'}${formatter.format(Math.abs(factor.contribution))}` : formatter.format(factor.abs_importance ?? Math.abs(factor.coefficient ?? 0))}</strong></div>)}</section>}
          {comparables.length > 0 && <section className="recent-card"><h2>Similar historical sales</h2><p className="factor-note">Nearest privacy-safe King County Assessor records—not verified active listings or appraisals.</p>{comparables.map((property, index) => <div className="comparable-item" key={`${property.zip_code}-${property.price}-${index}`}><div><strong>{formatter.format(property.price)}</strong><span>{property.beds} bd · {property.baths} ba · {Number(property.size).toLocaleString()} sq ft</span></div><div><span>ZIP {property.zip_code}</span><small>{Math.round(property.similarity * 100)}% similar</small></div></div>)}</section>}
          {recent.length > 0 && <section className="recent-card"><h2>Recent estimates</h2>{recent.map((item, index) => <div className="recent-item" key={`${item.date}-${index}`}><span>{item.profile ?? 'Property profile'} · {item.date}</span><strong>{formatter.format(item.value)}</strong></div>)}</section>}
        </aside>
      </div>
    </main>
  );
}

export default Houseprice;
