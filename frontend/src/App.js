// src/App.js
import React from 'react';
import { Route, Routes } from 'react-router-dom';  // Only import Routes and Route here
import Dashboard from './pages/Dashboard';
import Houseprice from './pages/Houseprice';
import Navbar from './pages/NavBar';
import About from './pages/About';
import HarrisCounty from './pages/HarrisCounty';

import './App.css';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/house-price" element={<Houseprice />} />
        <Route path="/about" element={<About />} />
        <Route path="/harris-county" element={<HarrisCounty />} />
      </Routes>
    </div>
  );
}

export default App;
