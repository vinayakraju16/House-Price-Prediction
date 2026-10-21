/**
 * Form fields, sample homes and what-if controls for each market's model.
 * Field names match the model inputs exported for that market.
 */
const currentYear = new Date().getFullYear();

const zipField = (hint) => ({ name: 'zip_code', label: 'ZIP code', hint, step: 1, group: 'Location' });
const ageFields = [
  { name: 'year_built', label: 'Year built', min: 1800, max: currentYear, step: 1, group: 'Age and renovation', required: false },
  { name: 'year_renovated', label: 'Year renovated', hint: 'Leave blank if never renovated', min: 1800, max: currentYear, step: 1, group: 'Age and renovation', required: false },
];

const kingCountyGrades = { 6: 'Low', 7: 'Average', 8: 'Good', 9: 'Better', 10: 'Very good', 11: 'Excellent', 12: 'Luxury', 13: 'Exceptional' };
const kingCountyConditions = { 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Very good' };

// Wake County grades are letters with +/- percentage modifiers; the model uses
// AA=6, A=5, B=4, C=3, D=2 plus the modifier in hundredths (C+10 = 3.10).
const wakeGrades = [
  ['6', 'Excellent / custom (AA)'], ['5', 'High (A)'], ['4', 'Above average (B)'],
  ['3.1', 'Average-plus (C+10, most new builds)'], ['3', 'Average (C)'], ['2', 'Below average (D)'],
];
// Accrued condition, "percent good" in the county record (median 82).
const wakeConditions = [['95', 'Like new'], ['88', 'Very good'], ['82', 'Good'], ['75', 'Fair'], ['60', 'Worn']];
const wakeStyles = ['Conventional', 'Townhouse', 'Ranch', 'Split Level', 'Split Foyer', 'Contemporary', 'MODULAR'];
const wakeStyleLabels = { MODULAR: 'Modular' };
const wakeWalls = [['A', 'Wood / fiber-cement siding'], ['Z', 'Vinyl or aluminum siding'], ['B', 'Brick'], ['D', 'Brick and siding'], ['G', 'Stucco'], ['I', 'Stone']];

export const MARKET_FORMS = {
  'king-county': {
    placeLabel: 'Seattle',
    fields: [
      zipField('Seattle ZIP 98101–98199 or a Wake County, NC ZIP'),
      { name: 'beds', label: 'Bedrooms', hint: 'Number of bedrooms', min: 1, step: 1, group: 'Property basics' },
      { name: 'baths', label: 'Bathrooms', hint: 'Number of bathrooms', min: 0.5, step: 0.5, group: 'Property basics' },
      { name: 'size', label: 'Home size', hint: 'Square feet', min: 1, step: 1, group: 'Property basics' },
      { name: 'lot_size', label: 'Lot size', hint: 'Square feet; use 0 when unavailable', min: 0, step: 1, group: 'Property basics' },
      { name: 'stories', label: 'Stories', hint: 'Above-ground story count', min: 0.5, max: 10, step: 0.5, group: 'Building details', required: false },
      { name: 'grade', label: 'Building grade', hint: 'Construction quality recorded by the assessor', kind: 'select', group: 'Building details', required: false, options: Object.entries(kingCountyGrades).map(([value, text]) => [value, `${value} — ${text}`]) },
      { name: 'condition', label: 'Condition', hint: 'Current physical condition', kind: 'select', group: 'Building details', required: false, options: Object.entries(kingCountyConditions) },
      ...ageFields,
      { name: 'finished_basement_sqft', label: 'Finished basement', hint: 'Finished basement square feet', min: 0, step: 1, group: 'Additional spaces', required: false },
      { name: 'garage_sqft', label: 'Garage area', hint: 'Attached and basement garage square feet', min: 0, step: 1, group: 'Additional spaces', required: false },
      { name: 'fireplaces', label: 'Fireplaces', min: 0, max: 30, step: 1, group: 'Additional spaces', required: false },
      { name: 'has_view', label: 'Scenic view', hint: 'Whether the property has a recorded view', kind: 'select', group: 'Additional spaces', required: false, options: [['0', 'No'], ['1', 'Yes']] },
    ],
    sample: {
      zip_code: '98144', beds: '3', baths: '2.5', size: '2590', lot_size: '6000',
      stories: '2', grade: '8', condition: '3', year_built: '1996', year_renovated: '',
      finished_basement_sqft: '450', garage_sqft: '400', fireplaces: '1', has_view: '0',
    },
    profile: (data) => `${data.beds} bd / ${data.baths} ba`,
    sliders: [
      { name: 'size', label: 'Living area', min: 400, max: 8000, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
      { name: 'grade', label: 'Building grade', min: 6, max: 13, step: 1, format: (v) => `${v} — ${kingCountyGrades[v] ?? ''}` },
      { name: 'condition', label: 'Condition', min: 1, max: 5, step: 1, format: (v) => kingCountyConditions[v] ?? v },
      { name: 'finished_basement_sqft', label: 'Finished basement', min: 0, max: 2500, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
      { name: 'garage_sqft', label: 'Garage area', min: 0, max: 1200, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
    ],
    toggles: [{ name: 'has_view', label: 'Scenic view' }],
  },
  'wake-county': {
    placeLabel: 'Wake County',
    fields: [
      zipField('Wake County, NC ZIP, e.g. 27615'),
      { name: 'baths', label: 'Bathrooms', hint: 'Number of bathrooms', min: 0.5, step: 0.5, group: 'Property basics' },
      { name: 'size', label: 'Home size', hint: 'Heated square feet', min: 1, step: 1, group: 'Property basics' },
      { name: 'lot_size', label: 'Lot size', hint: 'Square feet (1 acre = 43,560); 0 for condos', min: 0, step: 1, group: 'Property basics' },
      { name: 'property_type', label: 'Home style', kind: 'select', group: 'Building details', required: false, options: wakeStyles.map((style) => [style, wakeStyleLabels[style] ?? style]) },
      { name: 'stories', label: 'Stories', kind: 'select', group: 'Building details', required: false, options: [['1', '1'], ['1.5', '1½'], ['1.75', '1¾'], ['2', '2'], ['3', '3']] },
      { name: 'grade', label: 'Construction grade', hint: 'Quality grade in the county record', kind: 'select', group: 'Building details', required: false, options: wakeGrades },
      { name: 'condition', label: 'Condition', kind: 'select', group: 'Building details', required: false, options: wakeConditions },
      { name: 'exterior_wall', label: 'Exterior', kind: 'select', group: 'Building details', required: false, options: wakeWalls },
      ...ageFields,
      { name: 'has_basement', label: 'Basement', kind: 'select', group: 'Additional spaces', required: false, options: [['0', 'No'], ['1', 'Yes']] },
      { name: 'has_central_air', label: 'Central air', kind: 'select', group: 'Additional spaces', required: false, options: [['1', 'Yes'], ['0', 'No']] },
    ],
    sample: {
      zip_code: '27615', baths: '2.5', size: '2200', lot_size: '10890', property_type: 'Conventional',
      stories: '2', grade: '3.1', condition: '82', exterior_wall: 'Z', year_built: '2005',
      year_renovated: '', has_basement: '0', has_central_air: '1',
    },
    profile: (data) => `${data.baths} ba / ${Number(data.size).toLocaleString()} sq ft`,
    sliders: [
      { name: 'size', label: 'Living area', min: 600, max: 6500, step: 10, format: (v) => `${Number(v).toLocaleString()} sq ft` },
      { name: 'grade', label: 'Construction grade', steps: [2, 3, 3.1, 4, 5, 6], format: (v) => (wakeGrades.find(([value]) => Number(value) === Number(v))?.[1] ?? v) },
      { name: 'condition', label: 'Condition (percent good)', min: 50, max: 100, step: 1, format: (v) => `${v}%` },
      { name: 'baths', label: 'Bathrooms', min: 1, max: 3.5, step: 0.5, format: (v) => `${v}` },
    ],
    toggles: [{ name: 'has_basement', label: 'Basement' }],
  },
};

export const ALL_FIELD_NAMES = [...new Set(Object.values(MARKET_FORMS).flatMap((form) => form.fields.map(({ name }) => name)))];
