/**
 * Application Configuration
 * Centralized API base URL supporting environment variable overrides (VITE_API_URL).
 */
export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
