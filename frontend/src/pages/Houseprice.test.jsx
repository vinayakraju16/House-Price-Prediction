import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import Houseprice from './Houseprice';

const response = {
  prediction: 958996.03,
  range: { low: 644476.95, high: 1424328.24, nominal_coverage: 0.9 },
  confidence: { level: 'High', message: 'Inputs are within typical training ranges.' },
  disclaimer: 'This estimate is not a formal appraisal.',
  top_factors: [
    { feature: 'zip_code', label: 'ZIP code', contribution: -69470 },
    { feature: 'size', label: 'Home size', contribution: 67884 },
  ],
  comparables: [
    { price: 925000, beds: 3, baths: 2.5, size: 2500, zip_code: 98144, similarity: 0.92 },
  ],
};

beforeEach(() => {
  localStorage.clear();
  global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response });
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('renders a calibrated estimate, local factors, and comparable records', async () => {
  render(<Houseprice />);

  fireEvent.change(screen.getByLabelText(/Bedrooms/i), { target: { value: '3' } });
  fireEvent.change(screen.getByLabelText(/Bathrooms/i), { target: { value: '2.5' } });
  fireEvent.change(screen.getByLabelText(/Home size/i), { target: { value: '2590' } });
  fireEvent.change(screen.getByLabelText(/^Lot size/i), { target: { value: '6000' } });
  fireEvent.change(screen.getByLabelText(/^ZIP code/i), { target: { value: '98144' } });
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByText('90% empirical range')).toBeInTheDocument();
  expect(screen.getAllByText('$958,996')).toHaveLength(2);
  expect(screen.getByRole('heading', { name: /What shaped this estimate/i })).toBeInTheDocument();
  expect(screen.getByText('−$69,470')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /Similar historical sales/i })).toBeInTheDocument();
  expect(screen.getByText('92% similar')).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringMatching(/\/predict\/$/),
    expect.objectContaining({ method: 'POST' }),
  );
});

test('renders a structured API validation message', async () => {
  global.fetch.mockResolvedValueOnce({
    ok: false,
    json: async () => ({
      error: 'validation_error',
      message: 'zip_code: Must be a valid 5-digit ZIP code.',
      details: { zip_code: 'Must be a valid 5-digit ZIP code.' },
    }),
  });
  render(<Houseprice />);

  fireEvent.click(screen.getByRole('button', { name: /Use sample/i }));
  fireEvent.click(screen.getByRole('button', { name: /Calculate estimate/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(
    'zip_code: Must be a valid 5-digit ZIP code.',
  );
});
