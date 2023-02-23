import React, { useMemo, useContext, createContext, useState } from 'react';

export const AnalystOutcomeContext = createContext({});

export const useAnalystOutcome = () => {
    const context = useContext(AnalystOutcomeContext);
    if (context === undefined) {
        throw new Error('useAnalystOutcome be used within a AnalystOutcomeContextProvider');
    }
    return context;
};

export const AnalystOutcomeContextProvider = ({ children }) => {
    const [approved, setApproved] = useState([]);
    const [dismissed, setDismissed] = useState([]);

    const context = useMemo(() => ({
        approved, setApproved,
        dismissed, setDismissed
    }), [approved, dismissed]);

    return <AnalystOutcomeContext.Provider value={context}>{ children }</AnalystOutcomeContext.Provider>;
};
