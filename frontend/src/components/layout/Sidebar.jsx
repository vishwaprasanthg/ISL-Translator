import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ROUTES, APP_NAME } from '../../utils/constants';

export default function Sidebar() {
  const location = useLocation();

  return (
    <div className="w-64 bg-bg-secondary border-r border-gray-800 flex flex-col h-full flex-shrink-0">
      <div className="p-6">
        <h1 className="text-lg font-bold text-white flex items-center space-x-2 leading-tight">
          <span className="text-2xl">🤟</span>
          <span>{APP_NAME.split(' ').slice(1, 4).join(' ')}</span>
        </h1>
      </div>
      
      <nav className="flex-1 px-4 py-2 space-y-1 overflow-y-auto">
        {ROUTES.slice(0, 5).map((route) => {
          const isActive = location.pathname === route.path;
          const Icon = route.icon;
          
          return (
            <Link
              key={route.path}
              to={route.path}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg font-medium transition-colors ${
                isActive 
                  ? 'bg-accent-primary/10 text-accent-primary' 
                  : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{route.name}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-800 space-y-1">
        {ROUTES.slice(5).map((route) => {
          const isActive = location.pathname === route.path;
          const Icon = route.icon;
          
          return (
            <Link
              key={route.path}
              to={route.path}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg font-medium transition-colors ${
                isActive 
                  ? 'bg-accent-primary/10 text-accent-primary' 
                  : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{route.name}</span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
