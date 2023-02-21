import React from 'react';
import Highlighter from 'react-highlight-words';
import { ApproveDismiss } from '../ApproveDismiss';

export const AnalyseText = ({ handleDismiss }) => {
    return <div>
        <p>
            Analysed content counter: 1
            <span className="ms-3">Analysed against: 1000 articles</span>
        </p>
        <h4>100% likelihood of misinformation</h4>
        <Highlighter searchWords={['nightmares']}
            textToHighlight={'drinking milk before bed gives you nightmares'}
            highlightClassName="p-0"
        />
        <div className="mt-2">
            <ApproveDismiss handleDismiss={handleDismiss} />
        </div>
        <hr className="my-4" />
    </div>
};
