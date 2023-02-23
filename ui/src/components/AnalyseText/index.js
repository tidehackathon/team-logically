import React from 'react';
import Highlighter from 'react-highlight-words';
import { ApproveDismiss } from '../ApproveDismiss';

export const AnalyseText = ({ text }) => {
    return <div>
        <p>
            Analysed content counter: <strong>1</strong>
            <span className="ms-3">Analysed against: <strong>1000 articles</strong></span>
        </p>
        <h4>100% likelihood of disinformation</h4>
        <Highlighter searchWords={['hoax']}
            textToHighlight={'the war in ukraine is a hoax'}
            highlightClassName="p-0"
        />
        <div className="mt-2">
            <ApproveDismiss data={{ id: text, claim: text, content: text, percentage: 100 }} />
        </div>
        <hr className="my-4" />
    </div>
};
