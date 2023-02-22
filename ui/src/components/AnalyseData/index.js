import React, { useState } from 'react';
import { Button, Table } from 'reactstrap';
import { ApproveDismiss } from '../ApproveDismiss';
import { Speedometer } from '../Speedometer';

export const AnalyseData = ({ data }) => {
    const [limit, setLimit] = useState(100);
    const [dismissed, setDismissed] = useState([]);
    return <div>
        <Table borderless style={{maxWidth: 800}}>
            <tbody>
                <tr>
                    <td className="px-0">Analysed content counter:</td>
                    <td className="px-0"><strong>{data.length}</strong></td>
                    <td></td>
                    <td className="px-0">Analysed against:</td>
                    <td className="px-0"><strong>1000000 articles</strong></td>
                </tr>
                <tr>
                    <td className="px-0">Claims found:</td>
                    <td className="px-0">{data.length}</td>
                </tr>
                <tr>
                    <td className="px-0">Likely disinformation:</td>
                    <td className="px-0">{data.filter(item => item.percentage > 40).length}</td>
                </tr>
                <tr>
                    <td className="px-0">Unlikely disinformation:</td>
                    <td className="px-0">{data.filter(item => item.percentage < 40).length}</td>
                </tr>
            </tbody>
        </Table>

        <div className="overflow-auto mt-2" style={{ maxHeight: 450 }}>
            <Table className="border">
                <thead className="sticky-top bg-white border-top">
                    <tr>
                        <td>Content</td>
                        <td>Claim</td>
                        <td style={{ width: 150 }}>Disinfo likelihood</td>
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
                                <td>
                                    <div className="d-flex">
                                        <p className="mb-0 me-2"><strong>{item.percentage}%</strong></p>
                                        <Speedometer value={item.percentage} />
                                    </div>
                                </td>
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
        <hr className="my-4" />
    </div>
};
