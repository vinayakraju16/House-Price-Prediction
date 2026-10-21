import comparables from './comparables-3.0.0.json';
import fixtures from './__fixtures__/parity-3.0.0.json';
import model from './king-county-3.0.0.json';
import {
  buildInputRow,
  estimate,
  explainPrediction,
  findComparables,
  PredictionInputError,
  predictRow,
  validatePayload,
} from './predictor';

// Fixtures come from `python -m ml.training.export_browser_model --fixtures`: random
// properties scored by the real scikit-learn pipeline and backend/api/services.py.
const fixtureDate = new Date(Date.UTC(fixtures.sale_year, fixtures.sale_month - 1, 15));
const sampleHome = {
  beds: '3', baths: '2.5', size: '2590', lot_size: '6000', zip_code: '98144',
  stories: '2', grade: '8', condition: '3', year_built: '1996', year_renovated: '',
  finished_basement_sqft: '450', garage_sqft: '400', fireplaces: '1', has_view: '0',
};

function validationError(payload) {
  try {
    validatePayload(payload, fixtureDate);
  } catch (error) {
    return error;
  }
  throw new Error('expected validation to fail');
}

describe('parity with the scikit-learn pipeline', () => {
  test.each(fixtures.cases.map((item, index) => [index, item]))('case %i prediction', (_, item) => {
    const row = buildInputRow(model, item.payload, fixtureDate);
    expect(Math.abs(predictRow(model, row) - item.prediction) / item.prediction).toBeLessThan(1e-9);
  });

  test('explanation factors match the backend and add up to the prediction', () => {
    let compared = 0;
    for (const item of fixtures.cases) {
      const row = buildInputRow(model, item.payload, fixtureDate);
      const explanation = explainPrediction(model, row);
      const total = explanation.top_factors.reduce((sum, factor) => sum + factor.contribution, 0);
      expect(explanation.base_value).toBeCloseTo(item.base_value, 4);
      expect(explanation.base_value + total).toBeCloseTo(item.prediction, 4);
      if (!item.explanation_consistent) continue;
      compared += 1;
      for (const factor of explanation.top_factors) {
        expect(factor.contribution).toBeCloseTo(item.contributions[factor.feature], 4);
      }
    }
    expect(compared).toBeGreaterThan(fixtures.cases.length * 0.9);
  });

  test('comparable sales match the backend', () => {
    for (const item of fixtures.cases) {
      const found = findComparables(model, comparables, buildInputRow(model, item.payload, fixtureDate));
      found.forEach((property, index) => {
        expect(property.similarity).toBeCloseTo(item.comparable_similarity[index], 10);
      });
      // numpy's argsort may order exactly tied records differently; the set is identical.
      expect(found.map((property) => property.price).sort()).toEqual([...item.comparable_prices].sort());
    }
  });
});

describe('estimate', () => {
  test('returns the same response shape as the former /predict/ endpoint', () => {
    const result = estimate(model, comparables, sampleHome, fixtureDate);
    const interval = model.prediction_interval;
    expect(result.prediction).toBeGreaterThan(0);
    expect(result.range.low).toBeCloseTo(result.prediction + interval.lower_residual_quantile, 1);
    expect(result.range.high).toBeCloseTo(result.prediction + interval.upper_residual_quantile, 1);
    expect(result.range.nominal_coverage).toBe(0.9);
    expect(result.confidence.level).toBe('High');
    expect(result.top_factors).toHaveLength(model.features.length);
    expect(result.comparables).toHaveLength(5);
    expect(result.model_version).toBe('3.0.0');
  });

  test('lowers confidence for values outside the training ranges', () => {
    const result = estimate(model, comparables, { ...sampleHome, beds: '9', size: '12000' }, fixtureDate);
    expect(result.confidence.level).toBe('Medium');
    expect(result.confidence.unusual_features).toEqual(['beds', 'size']);
  });
});

describe('validation mirrors backend/api/validation.py', () => {
  test('reports missing required fields', () => {
    const error = validationError({ beds: '3' });
    expect(error).toBeInstanceOf(PredictionInputError);
    expect(error.details).toEqual({
      baths: 'This field is required.',
      size: 'This field is required.',
      lot_size: 'This field is required.',
      zip_code: 'This field is required.',
    });
  });

  test('uses the backend messages for invalid values', () => {
    const error = validationError({ ...sampleHome, zip_code: '10001', beds: '2.5', lot_size: '6000000', size: 'big' });
    expect(error.details).toEqual({
      zip_code: 'Must be a Seattle-area ZIP code from 98101 to 98199.',
      beds: 'Must be a whole number.',
      lot_size: 'Must be between 0 and 5e+06.',
      size: 'Must be a number.',
    });
    expect(error.message).toContain('zip_code: Must be a Seattle-area ZIP code from 98101 to 98199.');
  });

  test('treats a renovation year of 0 as never renovated', () => {
    expect(validatePayload({ ...sampleHome, year_renovated: '0' }, fixtureDate)).not.toHaveProperty('year_renovated');
    expect(validationError({ ...sampleHome, year_renovated: '1990' }).details).toEqual({
      year_renovated: 'Cannot be earlier than year built.',
    });
  });
});
