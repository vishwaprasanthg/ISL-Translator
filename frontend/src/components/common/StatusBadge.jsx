import React from 'react';

export default function StatusBadge({ status, text }) {
  const getStatusColor = () => {
    switch (status) {
      case 'ready': return 'bg-accent-success/20 text-accent-success border border-accent-success/30';
      case 'loading': return 'bg-accent-warning/20 text-accent-warning border border-accent-warning/30';
      case 'error': return 'bg-accent-danger/20 text-accent-danger border border-accent-danger/30';
      case 'not_trained':
      default: return 'bg-gray-800 text-text-secondary border border-gray-700';
    }
  };

  return (
    <span className={`px-2 py-1 text-xs font-medium uppercase tracking-wider rounded-md flex items-center gap-2 ${getStatusColor()}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${status === 'ready' ? 'bg-accent-success' : status === 'loading' ? 'bg-accent-warning' : status === 'error' ? 'bg-accent-danger' : 'bg-gray-500'}`}></span>
      {text}
    </span>
  );
}
