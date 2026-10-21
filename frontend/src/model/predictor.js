/**
 * In-browser port of the Django prediction service (backend/api/services.py and
 * backend/api/validation.py) for the King County HistGradientBoosting model.
 *
 * The model parameters come from `python -m ml.training.export_browser_model`, which reads
 * the fitted scikit-learn pipeline directly. Every step below mirrors one step of that
 * pipeline with the same floating-point operations, so results match scikit-learn; the
 * parity test in predictor.test.js checks this against fixtures scored by Python.
 */

export const FEATURE_LABELS = {
  beds: 'Bedrooms',
  baths: 'Bathrooms',
  size: 'Living area',
  lot_size: 'Lot size',
  zip_code: 'ZIP code',
  stories: 'Stories',
  grade: 'Building grade',
  condition: 'Condition',
  year_built: 'Year built',
  year_renovated: 'Year renovated',
  finished_basement_sqft: 'Finished basement',
  garage_sqft: 'Garage area',
  fireplaces: 'Fireplaces',
  heat_system: 'Heating system',
  has_view: 'View',
  property_type: 'Property type',
  sale_year: 'Valuation year',
  sale_month: 'Valuation month',
};

export const DISCLAIMER = 'Experimental estimate based on King County Assessor sale records; not a formal appraisal.';

const REQUIRED_FEATURES = ['beds', 'baths', 'size', 'lot_size', 'zip_code'];
const OPTIONAL_FEATURES = [
  'stories', 'grade', 'condition', 'year_built', 'year_renovated',
  'finished_basement_sqft', 'garage_sqft', 'fireplaces', 'heat_system',
  'has_view', 'property_type',
];
const INTEGER_FEATURES = new Set([
  'beds', 'zip_code', 'grade', 'condition', 'year_built', 'year_renovated',
  'fireplaces', 'heat_system', 'has_view', 'property_type',
]);
const CATEGORICAL_FEATURES = new Set(['zip_code', 'heat_system', 'property_type']);

export class PredictionInputError extends Error {
  constructor(message, details) {
    super(message);
    this.name = 'PredictionInputError';
    this.details = details ?? { request: message };
  }
}

const isBlank = (value) => value === undefined || value === null || value === '';

/**
 * Python-style float(): returns null where Python raises (so the caller reports "Must be a
 * number."), and NaN/Infinity for the strings Python accepts as such.
 */
function toNumber(value) {
  if (typeof value === 'number') return value;
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (typeof value !== 'string') return null;
  const text = value.trim();
  if (/^[+-]?nan$/i.test(text)) return Number.NaN;
  if (/^[+-]?inf(inity)?$/i.test(text)) return text.startsWith('-') ? -Infinity : Infinity;
  const digits = '\\d(?:_?\\d)*';
  const pattern = new RegExp(`^[+-]?(?:${digits}(?:\\.(?:${digits})?)?|\\.${digits})(?:e[+-]?${digits})?$`, 'i');
  return pattern.test(text) ? Number(text.replaceAll('_', '')) : null;
}

/** Python's `format(value, 'g')`, used in the backend's range messages (5000000 -> 5e+06). */
function formatG(value) {
  if (value === 0) return '0';
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  if (exponent < -4 || exponent >= 6) {
    const [mantissa, power] = value.toExponential(5).split('e');
    const sign = power.startsWith('-') ? '-' : '+';
    return `${mantissa.replace(/\.?0+$/, '')}e${sign}${power.replace(/^[+-]/, '').padStart(2, '0')}`;
  }
  return String(Number(value.toPrecision(6)));
}

function validateZipCode(value, details) {
  const numeric = toNumber(value);
  if (numeric === null) {
    details.zip_code = 'Must be a valid 5-digit ZIP code.';
    return null;
  }
  if (!Number.isFinite(numeric) || !Number.isInteger(numeric)) {
    details.zip_code = 'Must be a whole 5-digit ZIP code.';
    return null;
  }
  if (numeric < 10000 || numeric > 99999) {
    details.zip_code = 'Must be a valid 5-digit ZIP code.';
    return null;
  }
  return numeric;
}

function validateNumericFeature(name, value, details) {
  const numeric = toNumber(value);
  if (numeric === null) {
    details[name] = 'Must be a number.';
  } else if (!Number.isFinite(numeric)) {
    details[name] = 'Must be a finite number.';
  } else if (name === 'size' && numeric <= 0) {
    details[name] = 'Must be greater than zero.';
  } else if (numeric < 0) {
    details[name] = 'Cannot be negative.';
  } else if (INTEGER_FEATURES.has(name) && !Number.isInteger(numeric)) {
    details[name] = 'Must be a whole number.';
  } else {
    return numeric;
  }
  return null;
}

function validateFeatureRanges(values, details, currentYear) {
  const ranges = {
    beds: [0, 20], baths: [0, 20], size: [1, 30_000],
    lot_size: [0, 5_000_000], stories: [0.5, 100], grade: [1, 13],
    condition: [1, 5], year_built: [1800, currentYear],
    year_renovated: [0, currentYear], finished_basement_sqft: [0, 30_000],
    garage_sqft: [0, 30_000], fireplaces: [0, 30], heat_system: [0, 99],
    has_view: [0, 1], property_type: [0, 999],
  };
  for (const [name, [minimum, maximum]] of Object.entries(ranges)) {
    if (name in values && !(minimum <= values[name] && values[name] <= maximum)) {
      details[name] = `Must be between ${formatG(minimum)} and ${formatG(maximum)}.`;
    }
  }
  if ('zip_code' in values && !(values.zip_code >= 98101 && values.zip_code <= 98199)) {
    details.zip_code = 'Must be a Seattle-area ZIP code from 98101 to 98199.';
  }
  if (values.year_renovated === 0) delete values.year_renovated;
  if ('year_renovated' in values && 'year_built' in values && values.year_renovated < values.year_built) {
    details.year_renovated = 'Cannot be earlier than year built.';
  }
}

/** Port of validate_prediction_payload: same rules, same messages. */
export function validatePayload(payload, now = new Date()) {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    const message = 'The request body must be a JSON object.';
    throw new PredictionInputError(message, { body: message });
  }
  const details = {};
  const values = {};
  for (const name of REQUIRED_FEATURES) {
    const raw = payload[name];
    if (isBlank(raw)) {
      details[name] = 'This field is required.';
      continue;
    }
    const normalized = name === 'zip_code' ? validateZipCode(raw, details) : validateNumericFeature(name, raw, details);
    if (normalized !== null) values[name] = normalized;
  }
  for (const name of OPTIONAL_FEATURES) {
    const raw = payload[name];
    if (isBlank(raw)) continue;
    const normalized = validateNumericFeature(name, raw, details);
    if (normalized !== null) values[name] = normalized;
  }
  validateFeatureRanges(values, details, now.getUTCFullYear());
  const problems = Object.entries(details);
  if (problems.length) {
    throw new PredictionInputError(problems.map(([field, problem]) => `${field}: ${problem}`).join('; '), details);
  }
  return values;
}

/** Port of build_input_frame: training defaults, then validated values, then today's date. */
export function buildInputRow(model, payload, now = new Date()) {
  const values = validatePayload(payload, now);
  const row = {};
  for (const name of model.features) row[name] = model.feature_defaults[name] ?? null;
  Object.assign(row, values);
  if (model.features.includes('sale_year')) row.sale_year = now.getUTCFullYear();
  if (model.features.includes('sale_month')) row.sale_month = now.getUTCMonth() + 1;
  return row;
}

const asNumber = (value) => (value === null || value === undefined ? Number.NaN : Number(value));
const finiteOrNaN = (value) => (Number.isFinite(value) ? value : Number.NaN);

/** Port of KingCountyFeatureEngineer.transform for a single row. */
export function engineerFeatures(row) {
  const out = {};
  for (const [name, value] of Object.entries(row)) {
    out[name] = CATEGORICAL_FEATURES.has(name) ? (value === null ? null : String(value)) : asNumber(value);
  }
  const { size, beds, baths, lot_size: lot, year_built: built, year_renovated: renovated, sale_year: saleYear } = out;
  out.property_age = Number.isNaN(saleYear - built) ? Number.NaN : Math.max(saleYear - built, 0);
  out.was_renovated = Number.isFinite(renovated) ? 1 : 0;
  out.years_since_renovation = Number.isFinite(renovated) ? finiteOrNaN(Math.max(saleYear - renovated, 0)) : Number.NaN;
  out.sqft_per_bedroom = beds > 0 ? size / beds : Number.NaN;
  out.bathroom_bedroom_ratio = beds > 0 ? baths / beds : Number.NaN;
  out.living_lot_ratio = lot > 0 ? Math.min(Math.max(size / lot, 0), 20) : Number.NaN;
  out.total_rooms = beds + baths;
  return out;
}

/** Imputation, standard scaling, and one-hot encoding from the fitted ColumnTransformer. */
export function transformRow(model, row) {
  const engineered = engineerFeatures(row);
  const { numeric, categorical } = model.preprocessing;
  const vector = [];
  numeric.columns.forEach((column, index) => {
    let value = engineered[column];
    if (Number.isNaN(value)) value = numeric.impute[index];
    vector.push((value - numeric.mean[index]) / numeric.scale[index]);
  });
  categorical.columns.forEach((column, index) => {
    const value = engineered[column] ?? categorical.impute[index];
    for (const category of categorical.categories[index]) vector.push(value === category ? 1 : 0);
  });
  return vector;
}

function rawPrediction(regressor, vector) {
  let raw = regressor.baseline;
  for (const tree of regressor.trees) {
    const { feature, threshold, left, right, value, missing_left: missingLeft } = tree;
    let node = 0;
    while (feature[node] !== -1) {
      const x = vector[feature[node]];
      const goLeft = Number.isNaN(x) ? missingLeft[node] === 1 : x <= threshold[node];
      node = goLeft ? left[node] : right[node];
    }
    raw += value[node];
  }
  return raw;
}

/** Full pipeline prediction in dollars for an already-built input row. */
export function predictRow(model, row) {
  return Math.expm1(rawPrediction(model.regressor, transformRow(model, row)));
}

/** Port of explain_prediction: deterministic permutation Shapley values. */
export function explainPrediction(model, row) {
  const reference = model.explanation_reference;
  if (!reference) return { method: 'unavailable', top_factors: [] };
  const names = model.features;
  const { permutations } = model.shapley;
  const totals = new Array(names.length).fill(0);
  let baseSum = 0;
  let finalSum = 0;
  for (const order of permutations) {
    const pathRow = {};
    for (const name of names) pathRow[name] = name in reference ? reference[name] : row[name];
    let previous = predictRow(model, pathRow);
    baseSum += previous;
    for (const index of order) {
      pathRow[names[index]] = row[names[index]];
      const next = predictRow(model, pathRow);
      totals[index] += next - previous;
      previous = next;
    }
    finalSum += previous;
  }
  const factors = names.map((feature, index) => {
    const contribution = totals[index] / permutations.length;
    return {
      feature,
      label: FEATURE_LABELS[feature] ?? feature.replaceAll('_', ' '),
      value: row[feature],
      reference_value: reference[feature] ?? null,
      contribution,
      direction: contribution >= 0 ? 'increased' : 'decreased',
    };
  });
  factors.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return {
    method: `permutation Shapley values (${permutations.length} deterministic paths)`,
    base_value: baseSum / permutations.length,
    prediction: finalSum / permutations.length,
    reference,
    top_factors: factors,
  };
}

/** Port of find_comparable_properties: nearest historical assessor sales. */
export function findComparables(model, comparables, row, limit = 5) {
  const { columns, log_scales: logScales, rows } = comparables;
  const ranges = model.typical_feature_ranges;
  const distance = new Float64Array(rows);
  for (const feature of ['beds', 'baths']) {
    const bounds = ranges[feature];
    const scale = Math.max(bounds.p95 - bounds.p05, 1);
    const query = Number(row[feature]);
    const data = columns[feature];
    for (let i = 0; i < rows; i += 1) distance[i] += Math.abs(data[i] - query) / scale;
  }
  for (const [feature, weight] of [['size', 2], ['lot_size', 0.5]]) {
    const scale = Math.max(logScales[feature], 1);
    const query = Math.log1p(Number(row[feature]));
    const data = columns[feature];
    for (let i = 0; i < rows; i += 1) distance[i] += (weight * Math.abs(Math.log1p(data[i]) - query)) / scale;
  }
  const queryZip = Math.trunc(Number(row.zip_code));
  for (let i = 0; i < rows; i += 1) if (columns.zip_code[i] !== queryZip) distance[i] += 2;

  const nearest = [];
  for (let i = 0; i < rows; i += 1) {
    if (nearest.length < limit || distance[i] < distance[nearest[nearest.length - 1]]) {
      nearest.push(i);
      nearest.sort((a, b) => distance[a] - distance[b] || a - b);
      if (nearest.length > limit) nearest.pop();
    }
  }
  return nearest.map((i) => ({
    price: columns.price[i],
    beds: columns.beds[i],
    baths: columns.baths[i],
    size: columns.size[i],
    lot_size: columns.lot_size[i],
    zip_code: columns.zip_code[i],
    similarity: 1 / (1 + distance[i]),
    source: 'King County Assessor historical sale',
    sale_date: columns.sale_date[i],
    year_built: columns.year_built[i],
  }));
}

function inputConfidence(model, row) {
  const ranges = model.typical_feature_ranges;
  const unusual = REQUIRED_FEATURES.filter((name) => {
    const bounds = ranges[name];
    if (!bounds) return false;
    const value = Number(row[name]);
    return !((bounds.p05 ?? bounds.minimum) <= value && value <= (bounds.p95 ?? bounds.maximum));
  });
  if (unusual.length === 0) return { level: 'High', message: 'All entered values are within typical ranges seen during training.', unusual_features: unusual };
  if (unusual.length <= 2) return { level: 'Medium', message: 'Some entered values are uncommon in the training data.', unusual_features: unusual };
  return { level: 'Low', message: 'Several entered values are outside typical training ranges.', unusual_features: unusual };
}

const round2 = (value) => Math.round(value * 100) / 100;

/** Port of estimate_details plus the /predict/ response shaping in views.py. */
export function estimate(model, comparables, payload, now = new Date()) {
  const row = buildInputRow(model, payload, now);
  const prediction = predictRow(model, row);
  const interval = model.prediction_interval;
  const low = Math.max(0, prediction + interval.lower_residual_quantile);
  const high = prediction + interval.upper_residual_quantile;
  const range = {
    low: round2(low),
    high: round2(high),
    margin: round2(Math.max(prediction - low, high - prediction)),
    method: interval.method,
    nominal_coverage: round2(interval.nominal_coverage),
  };
  const explanation = explainPrediction(model, row);
  return {
    prediction: round2(prediction),
    range,
    confidence: inputConfidence(model, row),
    model_version: model.model_version,
    model_factors: explanation.top_factors,
    top_factors: explanation.top_factors,
    explanation,
    comparables: findComparables(model, comparables, row),
    disclaimer: DISCLAIMER,
  };
}
