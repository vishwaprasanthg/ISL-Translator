import React from 'react';

export default function Card({ children, className = '', title }) {
  return (
    <div className={`bg-bg-secondary p-6 rounded-xl border border-gray-800 shadow-sm flex flex-col ${className}`}>
      {title && <h3 className="text-lg font-medium text-text-primary mb-4">{title}</h3>}
      {children}
    </div>
  );
}
