import React from 'react';
import { Link } from 'react-router-dom';
import { FaArrowRight, FaChartLine, FaClock, FaShieldAlt } from 'react-icons/fa';
import './All.css';
import houseImage from '../assets/images/houseimg1.jpg';

const Dashboard = () => (
  <main>
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">A clearer view of home value</p>
        <h1>Know what your home could be worth.</h1>
        <p className="hero-text">Get a data-informed price estimate from the details that make a property unique.</p>
        <div className="hero-actions">
          <Link to="/house-price" className="button button-primary">Get an estimate <FaArrowRight /></Link>
          <Link to="/about" className="text-link">See how it works</Link>
        </div>
      </div>
      <div className="hero-image-wrap">
        <img src={houseImage} alt="Contemporary house exterior" className="hero-image" />
        <div className="hero-float-card"><span>Typical estimate time</span><strong>Under a minute</strong></div>
      </div>
    </section>
    <section className="benefits section-shell">
      <article><span className="icon-badge"><FaChartLine /></span><h2>Data-informed</h2><p>Our model evaluates 14 meaningful property details together.</p></article>
      <article><span className="icon-badge"><FaClock /></span><h2>Fast to use</h2><p>Use a sample profile or enter your own home details.</p></article>
      <article><span className="icon-badge"><FaShieldAlt /></span><h2>Private by design</h2><p>We only use the values needed to calculate an estimate.</p></article>
    </section>
  </main>
);

export default Dashboard;
