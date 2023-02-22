import React, { useState } from 'react'
import { Button, Col, Row } from 'reactstrap';
import { AnalyseData } from './components/AnalyseData';
import { AnalyseText } from './components/AnalyseText';
import { DateHistogram } from './components/DateHistogram';
import { EmptyScreen } from './components/EmptyScreen';
import { FileUpload } from './components/FileUpload';
import { SingularInput } from './components/SingularInput';
import './variables.scss';

export const App = () => {
    const [dataset, setDataset] = useState(false);
    const [textInput, setTextInput] = useState('');
    const [fileInput, setFileInput] = useState([]);
    return <div className="p-4">
        <h1 className="text-center mb-4">NODDY: Networked Disinformation detection system</h1>
        <Row className="justify-content-center">
            <Col xs={12} lg={6} xl={4}>
                <SingularInput onChange={setTextInput} />
            </Col>
            <Col xs={12} lg={6} xl="auto">
                <FileUpload onChange={(data) => setFileInput(data.map((item) => {
                    const text = item.headlines || (item.title === 'Comment' ? item.body : item.title) || item.content || '';
                    return {
                        content: text,
                        claim: text,
                        percentage: Math.floor(Math.random() * 100) + 1
                    }
                }).sort((a, b) => b.percentage - a.percentage))} />
            </Col>
        </Row>
        <hr className="my-4" />
        {textInput && <AnalyseText text={textInput} handleDismiss={() => setTextInput('')} />}
        {fileInput.length !== 0 && <AnalyseData data={fileInput} />}
        {(!textInput && !fileInput.length) && <EmptyScreen />}
        <hr className="my-4" />
        <Button onClick={() => setDataset(!dataset)}>{dataset ? 'Hide' : 'Show'} dataset</Button>
        {dataset && <DateHistogram />}
    </div>
};
