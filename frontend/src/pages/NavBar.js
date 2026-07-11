import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { FaBars, FaHome, FaTimes } from 'react-icons/fa';

const Navbar = () => {
  const [open, setOpen] = useState(false);
  const closeMenu = () => setOpen(false);

  return (
    <header className="site-header">
      <nav className="navbar" aria-label="Main navigation">
        <NavLink to="/" className="brand" onClick={closeMenu}>
          <span className="brand-mark"><FaHome /></span>
          <span>Haven<span>Value</span></span>
        </NavLink>
        <button className="menu-toggle" onClick={() => setOpen(!open)} aria-label="Toggle menu" aria-expanded={open}>
          {open ? <FaTimes /> : <FaBars />}
        </button>
        <div className={`nav-links ${open ? 'is-open' : ''}`}>
          <NavLink to="/" end onClick={closeMenu}>Home</NavLink>
          <NavLink to="/house-price" onClick={closeMenu}>Estimate</NavLink>
          <NavLink to="/harris-county" onClick={closeMenu}>Texas estimate</NavLink>
          <NavLink to="/about" onClick={closeMenu}>How it works</NavLink>
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
