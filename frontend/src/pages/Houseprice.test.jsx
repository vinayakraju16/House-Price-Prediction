import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import comparables from '../model/comparables-3.0.0.json';
import model from '../model/king-county-3.0.0.json';
import { estimate } from '../model/predictor';
import Houseprice from './Houseprice';

const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const basics = { beds: '3', baths: '2.5', size: '2590', lot_size: '6000', zip_code: '98144' };

beforeEach(() => {
  localStorage.clear();
  // The estimate is computed in the browser, so no network request should be made.
  global.fetch = vi.fn();
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('calculates an estimate, local factors, and comparable records in the browser', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/Bedrooms/i), { target: { value: basics.beds } });
  fireEvent.change(screen.getByLabelText(/Bathrooms/i), { target: { value: basics.baths } });
  fireEvent.change(screen.getByLabelText(/Home size/i), { target: { value: basics.size } });
  fireEvent.change(screen.getByLabelText(/^Lot size/i), { target: { value: basics.lot_size } });
  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: basics.zip_code } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByText('90% empirical range', {}, { timeout: 15000 })).toBeInTheDocument();
  const expected = estimate(model, comparables, basics, new Date());
  expect(screen.getAllByText(money.format(expected.prediction))).toHaveLength(2);
  expect(screen.getByRole('heading', { name: /What shaped this estimate/i })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /Similar historical sales/i })).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalled();
}, 20000);

test('shows the same validation message the API used to return', async () => {
  render(<Houseprice />);

  fireEvent.click(screen.getByRole('button', { name: /Use sample/i }));
  fireEvent.change(screen.getByLabelText(/Bedrooms/i), { target: { value: '25' } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByRole('alert', {}, { timeout: 15000 })).toHaveTextContent('beds: Must be between 0 and 20.');
}, 20000);
