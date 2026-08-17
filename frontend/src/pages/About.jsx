import { useEffect, useState } from 'react';
import { FaCheckCircle } from 'react-icons/fa';
import { API_URL } from '../config';
import './All.css';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const About = () => {
  const [model, setModel] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/model-info/`).then((response) => response.ok ? response.json() : null)
      .then(setModel).catch(() => setModel(null));
  }, []);

  return (
    <main className="section-shell about-page">
      <p className="eyebrow">How it works</p>
      <h1>A practical starting point for your home search.</h1>
      <p className="lead">HavenValue uses a leakage-safe model trained on privacy-safe King County Assessor sales, building, and parcel records. It considers location, size, age, quality, condition, renovation, basement, garage, fireplaces, and view information when those details are available.</p>
      {model && <section className="model-summary" aria-label="Model performance"><div><span>Active version</span><strong>{model.model_version}</strong></div><div><span>Model</span><strong>{model.deployed_model_type}</strong></div><div><span>Training records</span><strong>{model.dataset_rows.toLocaleString()}</strong></div><div><span>Holdout MAE</span><strong>{money.format(model.metrics.mae)}</strong></div><div><span>Holdout RMSE</span><strong>{money.format(model.metrics.rmse)}</strong></div><div><span>Holdout R²</span><strong>{model.metrics.r2.toFixed(3)}</strong></div><div><span>Trained</span><strong>{model.training_date ? new Date(model.training_date).toLocaleDateString() : 'Unavailable'}</strong></div><div><span>Training price median</span><strong>{money.format(model.target_summary.median)}</strong></div></section>}
      <div className="about-grid">
        <section className="info-card"><h2>What to expect</h2><ul><li><FaCheckCircle /> An experimental estimate, not an appraisal</li><li><FaCheckCircle /> A range based on measured model error</li><li><FaCheckCircle /> Confidence that reflects whether values fit the training data</li></ul></section>
        <section className="info-card"><h2>Important note</h2><p>The comparable records are historical assessor sales, not active listings or verified appraisal comparables. Current market conditions and unrecorded property changes can materially affect a sale price. Consult a qualified local professional for an official valuation.</p></section>
      </div>
    </main>
  );
};

export default About;
