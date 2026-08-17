import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App';

test('renders the valuation dashboard', () => {
  render(<MemoryRouter><App /></MemoryRouter>);
  expect(screen.getByRole('heading', { name: /know what your home could be worth/i })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /get an estimate/i })).toHaveAttribute('href', '/house-price');
});
