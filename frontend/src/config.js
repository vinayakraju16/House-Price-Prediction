const configuredApiUrl = import.meta.env.VITE_API_URL;

export const API_URL = (configuredApiUrl || 'http://localhost:8000').replace(/\/$/, '');
