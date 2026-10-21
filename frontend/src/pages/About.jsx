import { useEffect, useState } from 'react';
import { FaCheckCircle } from 'react-icons/fa';
import { MARKETS, requestModelInfo } from '../model';
import './All.css';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

function ModelSummary({ name, model }) {
  const rows = model.training_rows ?? model.dataset_rows;
  return (
    <section className="model-summary" aria-label={`${name} model performance`}>
      <div><span>Market</span><strong>{name}</strong></div>
      <div><span>Active version</span><strong>{model.model_version}</strong></div>
      <div><span>Model</span><strong>{model.deployed_model_type ?? 'HistGradientBoosting, priced relative to ZIP home values'}</strong></div>
      <div><span>Training records</span><strong>{rows.toLocaleString()}</strong></div>
      <div><span>Holdout MAE</span><strong>{money.format(model.metrics.mae)}</strong></div>
      <div><span>Holdout RMSE</span><strong>{money.format(model.metrics.rmse)}</strong></div>
      <div><span>Holdout R²</span><strong>{model.metrics.r2.toFixed(3)}</strong></div>
      {model.metrics.median_ape !== undefined && <div><span>Median error</span><strong>{(model.metrics.median_ape * 100).toFixed(1)}%</strong></div>}
      <div><span>Trained</span><strong>{model.training_date ? new Date(model.training_date).toLocaleDateString() : 'Unavailable'}</strong></div>
    </section>
  );
}

const About = () => {
  const [models, setModels] = useState([]);

  useEffect(() => {
    let active = true;
    Promise.all(MARKETS.map((market) => requestModelInfo(market.id)
      .then((model) => ({ name: market.name, model }))
      .catch(() => null)))
      .then((loaded) => { if (active) setModels(loaded.filter(Boolean)); });
    return () => { active = false; };
  }, []);

  return (
    <main className="section-shell about-page">
      <p className="eyebrow">How it works</p>
      <h1>A practical starting point for your home search.</h1>
      <p className="lead">HavenValue runs one model per market and picks it from the ZIP code you enter. In Seattle, the model is trained on privacy-safe King County Assessor sales, building, and parcel records. In Wake County, NC, it is trained on the county’s appraiser-reviewed qualified sales and building records, and it prices each home relative to its ZIP’s Zillow home value index, so estimates follow the local market as it moves. Estimates consider location, size, age, quality, condition, and renovation, plus market-specific details when you provide them.</p>
      {models.map(({ name, model }) => <ModelSummary key={name} name={name} model={model} />)}
      <div className="about-grid">
        <section className="info-card"><h2>What to expect</h2><ul><li><FaCheckCircle /> An experimental estimate, not an appraisal</li><li><FaCheckCircle /> A range based on measured model error</li><li><FaCheckCircle /> Confidence that reflects whether values fit the training data</li><li><FaCheckCircle /> In Wake County, the latest ZIP market conditions next to your estimate</li></ul></section>
        <section className="info-card"><h2>Important note</h2><p>The comparable records are historical public-record sales, not active listings or verified appraisal comparables. Current market conditions and unrecorded property changes can materially affect a sale price. Consult a qualified local professional for an official valuation.</p></section>
      </div>
    </main>
  );
};

export default About;
