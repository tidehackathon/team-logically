import React, { useState } from 'react'
import { Button, Table } from 'reactstrap';
import { ApproveDismiss } from '../ApproveDismiss';
import { Speedometer } from '../Speedometer';

export const ClaimsTable = ({ data, keyword }) => {
    const [limit, setLimit] = useState(100);
    const [dismissed, setDismissed] = useState([]);
    return <div className={`overflow-auto ${keyword ? '' : 'mt-2 border'}`} style={{ maxHeight: keyword ? 550 :450 }}>
        <Table>
            <thead className="sticky-top bg-white" style={{ top: -1 }}>
                <tr>
                    <td className={keyword ? 'ps-4' : ''}>Content</td>
                    {!keyword && <td>Claim</td>}
                    <td style={{ width: 150 }}>Disinfo likelihood</td>
                    <td style={{ width: 200 }} className={keyword ? 'pr-4' : ''}>Action</td>
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => {
                    if (index > limit || dismissed.includes(index)) { return null; }
                    return (
                        <tr key={index}>
                            <td className={`dont-break-out ${keyword ? 'ps-4' : ''}`}>{item.content}</td>
                            {!keyword && <td className="dont-break-out">{item.claim}</td>}
                            <td>
                                <div className="d-flex">
                                    <p className="mb-0 me-2"><strong>{item.percentage}%</strong></p>
                                    <Speedometer value={item.percentage} />
                                </div>
                            </td>
                            <td className={keyword ? 'pr-4' : ''}>
                                <ApproveDismiss handleDismiss={() => setDismissed([ ...dismissed, index ])} />
                            </td>
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
};
