import React from 'react';
import EmptyState from '../components/common/EmptyState';
import { History as HistoryIcon } from 'lucide-react';

export default function History() {
  return (
    <div className="flex flex-col h-full gap-6">
      <h2 className="text-2xl font-bold">Translation History</h2>
      
      <div className="flex-1">
        <EmptyState 
          icon={HistoryIcon}
          title="No translation history yet"
          description="Translation history will appear here once the translator is active."
        />
      </div>
    </div>
  );
}
