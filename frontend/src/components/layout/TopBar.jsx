import React from 'react';
import { APP_NAME } from '../../utils/constants';
import StatusBadge from '../common/StatusBadge';
import { useAppContext } from '../../context/AppContext';

export default function TopBar() {
  const { modelStatus } = useAppContext();

  return (
    <header className="h-16 bg-bg-secondary/80 backdrop-blur border-b border-gray-800 flex items-center justify-between px-6 flex-shrink-0 z-10">
      <div className="flex items-center space-x-4">
        <h2 className="text-sm font-medium text-text-secondary hidden md:block">
          {APP_NAME}
        </h2>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <span className="text-xs text-text-secondary uppercase tracking-wider">Model Status</span>
          <StatusBadge 
            status={modelStatus} 
            text={modelStatus === 'not_trained' ? 'Not Trained' : modelStatus} 
          />
        </div>
      </div>
    </header>
  );
}
