import React, { useEffect, useState } from 'react';
import { FaCheckCircle } from 'react-icons/fa';
import './All.css';

const API_URL = process.env.REACT_APP_API_URL ?? 'http://localhost:8000';
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
      <p className="lead">HavenValue uses a Gradient Boosting model trained on a historical, selected-feature housing dataset. It considers fourteen property characteristics, including quality, lot size, age, rooms, basement, and garage details.</p>
      {model && <section className="model-summary" aria-label="Model performance"><div><span>Training records</span><strong>{model.dataset_rows.toLocaleString()}</strong></div><div><span>Held-out RMSE</span><strong>{money.format(model.metrics.rmse)}</strong></div><div><span>Held-out R²</span><strong>{model.metrics.r2}</strong></div><div><span>Training price median</span><strong>{money.format(model.target_summary.median)}</strong></div></section>}
      <div className="about-grid">
        <section className="info-card"><h2>What to expect</h2><ul><li><FaCheckCircle /> An experimental estimate, not an appraisal</li><li><FaCheckCircle /> A range based on measured model error</li><li><FaCheckCircle /> Confidence that reflects whether values fit the training data</li></ul></section>
        <section className="info-card"><h2>Important note</h2><p>This historical dataset does not contain current listings, local comparable sales, or live neighborhood prices. Local market conditions, renovations, property condition, and comparable sales can materially affect a real sale price. Consult a qualified local professional for an official valuation.</p></section>
      </div>
    </main>
  );
};

export default About;
