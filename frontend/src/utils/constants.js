import { Activity, Keyboard, Binary, BookOpen, History, Settings, Info } from 'lucide-react';

export const APP_NAME = 'AI-Powered Indian Sign Language Translator';

export const ROUTES = [
  { path: '/', name: 'Dashboard', icon: Activity },
  { path: '/alphabet', name: 'Alphabet', icon: Keyboard },
  { path: '/numbers', name: 'Numbers', icon: Binary },
  { path: '/vocabulary', name: 'Vocabulary', icon: BookOpen },
  { path: '/history', name: 'History', icon: History },
  { path: '/settings', name: 'Settings', icon: Settings },
  { path: '/about', name: 'About', icon: Info },
];

export const MODES = {
  ALPHABET: 'alphabet',
  NUMBERS: 'numbers',
  VOCABULARY: 'vocabulary',
};

export const MODEL_STATUS = {
  NOT_TRAINED: 'not_trained',
  LOADING: 'loading',
  READY: 'ready',
  ERROR: 'error',
};

export const DEFAULT_CONFIDENCE_THRESHOLD = 65;
