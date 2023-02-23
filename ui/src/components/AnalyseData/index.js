import React from 'react';
import { Table } from 'reactstrap';
import { ClaimsTable } from './ClaimsTable';
import { Engagement } from './Engagement';
import { KeywordWordcloud } from './KeywordWordcloud';

export const AnalyseData = ({ data }) => {
    return <div>
        <Table borderless style={{maxWidth: 800}}>
            <tbody>
                <tr>
                    <td className="px-0">Analysed content counter:</td>
                    <td className="px-0"><strong>{data.length}</strong></td>
                    <td></td>
                    <td className="px-0">Analysed against:</td>
                    <td className="px-0"><strong>6393253 articles</strong></td>
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

        <ClaimsTable data={data} />
        <div className="mt-4">
            <KeywordWordcloud data={data.slice(0, 500)} />
        </div>
        {data[0]?.engagement !== undefined && <div className="mt-4">
            <Engagement data={data} />
        </div>}
        <hr className="my-4" />
    </div>
};
