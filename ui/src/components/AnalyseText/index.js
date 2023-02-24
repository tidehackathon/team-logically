import React from 'react';
import Highlighter from 'react-highlight-words';
import { Spinner } from 'reactstrap';
import { ClaimsTable } from '../AnalyseData/ClaimsTable';
import { ApproveDismiss } from '../ApproveDismiss';

export const AnalyseText = ({ text, data, loading }) => {
    if (loading) {
        return <div className="p-5 d-flex align-items-center justify-content-center">
            <Spinner color="primary" />
        </div>
    }
    if (data.length) {
        return <div>
            <ClaimsTable data={data} />
        </div>
    }
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