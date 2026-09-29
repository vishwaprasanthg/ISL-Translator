import React from 'react';

export default function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="flex flex-col items-center justify-center p-8 h-full bg-bg-secondary rounded-xl border border-gray-800 border-dashed text-center">
      {Icon && <Icon className="w-12 h-12 text-gray-600 mb-4" />}
      <h3 className="text-lg font-medium text-text-primary">{title}</h3>
      <p className="text-sm text-text-secondary mt-2 max-w-sm">{description}</p>
    </div>
  );
}
