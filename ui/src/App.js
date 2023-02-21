import React, { useState } from 'react'
import { Col, Row } from 'reactstrap';
import { AnalyseData } from './components/AnalyseData';
import { AnalyseText } from './components/AnalyseText';
import { FileUpload } from './components/FileUpload';
import { SingularInput } from './components/SingularInput';
import './variables.scss';

export const App = () => {
    const [textInput, setTextInput] = useState('');
    const [fileInput, setFileInput] = useState([]);
    return <div className="px-4 pt-4">
        <Row>
            <Col xs={12} lg={6} xl={4}>
                <SingularInput onChange={setTextInput} />
            </Col>
            <Col xs={12} lg={6} xl={4}>
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
    </div>
};
