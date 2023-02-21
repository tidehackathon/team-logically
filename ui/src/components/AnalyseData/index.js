import React, { useState } from 'react';
import { Button, Table } from 'reactstrap';
import { ApproveDismiss } from '../ApproveDismiss';

export const AnalyseData = ({ data }) => {
    const [limit, setLimit] = useState(100);
    const [dismissed, setDismissed] = useState([]);
    return <div>
        <p>
            Analysed content counter: {data.length}
            <span className="ms-3">Analysed against: 1000000 articles</span>
        </p>
        <p>Claims found: {data.length}</p>
        <div className="overflow-auto mt-2" style={{ maxHeight: 450 }}>
            <Table>
                <thead className="sticky-top bg-white">
                    <tr>
                        <td>Content</td>
                        <td>Claim</td>
                        <td style={{ width: 150 }}>Misinfo likelihood</td>
                        <td style={{ width: 200 }}>Action</td>
                    </tr>
                </thead>
                <tbody>
                    {data.map((item, index) => {
                        if (index > limit || dismissed.includes(index)) { return null; }
                        return (
                            <tr key={index}>
                                <td className="dont-break-out">{item.content}</td>
                                <td className="dont-break-out">{item.claim}</td>
                                <td>{item.percentage}%</td>
                                <td><ApproveDismiss handleDismiss={() => setDismissed([ ...dismissed, index ])} /></td>
                            </tr>
                        )
                    })}
                    {limit < data.length && <tr>
                        <td className="border-0">
                            <Button onClick={() => setLimit(limit + 100)} color="primary">Load more</Button>
                        </td>
                    </tr>}
                </tbody>
            </Table>
        </div>
    </div>
};
