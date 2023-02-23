import moment from 'moment';
import React, { useState } from 'react'
import { Button, Table } from 'reactstrap';
import { ApproveDismiss } from '../ApproveDismiss';
import { Speedometer } from '../Speedometer';

export const ClaimsTable = ({ data, keyword, outcome }) => {
    const [limit, setLimit] = useState(100);
    const isModal = keyword || outcome;
    return <div className={`overflow-auto ${isModal ? '' : 'mt-2 border'}`} style={{ maxHeight: isModal ? 550 :450 }}>
        <Table>
            <thead className="sticky-top bg-white" style={{ top: -1 }}>
                <tr>
                    <td className={isModal ? 'ps-4' : ''}>Content</td>
                    {!isModal && <td>Claim</td>}
                    <td style={{ width: 150 }}>Disinfo likelihood</td>
                    <td style={{ width: outcome ? 150 : 200 }} className={isModal ? 'pr-4' : ''}>{outcome ? 'Outcome' : 'Action'}</td>
                    {outcome && <td style={{ width: 200 }}>Date of outcome</td>}
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => {
                    if (index > limit) { return null; }
                    return (
                        <tr key={index}>
                            <td className={`dont-break-out ${isModal ? 'ps-4' : ''}`}>{item.content}</td>
                            {!isModal && <td className="dont-break-out">
                                {item.claims.map(claim => <p key={claim}>{claim}</p>)}
                            </td>}
                            <td>
                                <div className="d-flex">
                                    <p className="mb-0 me-2"><strong>{item.percentage}%</strong></p>
                                    <Speedometer value={item.percentage} />
                                </div>
                            </td>
                            <td className={isModal ? 'pr-4' : ''}>
                                <ApproveDismiss data={item} />
                            </td>
                            {outcome && <td>{moment(item.dateOfOutcome).format('DD/MM/YYYY h:mm a')}</td>}
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
