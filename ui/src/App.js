import React, { useState } from 'react'
import { Button, Col, Row } from 'reactstrap';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { AnalyseData } from './components/AnalyseData';
import { AnalyseText } from './components/AnalyseText';
import { DateHistogram } from './components/DateHistogram';
import { EmptyScreen } from './components/EmptyScreen';
import { FileUpload } from './components/FileUpload';
import { SingularInput } from './components/SingularInput';
import './variables.scss';
import { useAnalystOutcome } from './AnalystOutcomeContext';
import { AnalystOutcomeModal } from './components/AnalystOutcomeModal';
import { useAPI } from './components/AnalyseText/useAPI';

export const App = () => {
    const [dataset, setDataset] = useState(false);
    const [textInput, setTextInput] = useState('');
    const [fileInput, setFileInput] = useState([]);
    const { approved, dismissed } = useAnalystOutcome();
    const { data, loading, callAPI } = useAPI();
    return <div className="p-4">
        <h1 className="text-center mb-4">NODDY: Networked Disinformation detection system</h1>
        <Row className="justify-content-center align-items-end">
            <Col xs={12} lg={6} xl={4}>
                <SingularInput onChange={(text) => { setFileInput([]); setTextInput(text); callAPI([{ content: text }]) }} />
            </Col>
            <Col xs={12} lg="auto">
                <FileUpload onChange={(data) => {
                    setTextInput('');
                    setFileInput(data.map((item, i) => {
                        const text = item.headlines || (item.title === 'Comment' ? item.body : item.title) || item.content;
                        if (!text) return null;
                        return {
                            id: item.id,
                            date: new Date(item.date || item.published || item.timestamp),
                            content: text,
                            claims: item.claims ? JSON.parse(item.claims) : [text],
                            percentage: item.score !== undefined ? item.score : Math.floor(Math.random() * 100) + 1,
                            ...(item.likeCount !== undefined ? {
                                engagement: parseInt(item.likeCount) + parseInt(item.replyCount || 0) + parseInt(item.retweetCount || 0)
                            } : {})
                        }
                    }).filter(a => a).sort((a, b) => b.percentage - a.percentage));
                }} />
            </Col>
            {(approved.length || dismissed.length) ? <Col xs="auto">
                <AnalystOutcomeModal />
            </Col> : null}
        </Row>
        <hr className="my-4" />
        {textInput && <AnalyseText text={textInput} data={data} loading={loading} />}
        {fileInput.length !== 0 && <AnalyseData data={fileInput} />}
        {(!textInput && !fileInput.length) && <EmptyScreen />}
        <Button onClick={() => setDataset(!dataset)}>{dataset ? 'Hide' : 'Show'} dataset</Button>
        {dataset && <DateHistogram />}
        <ToastContainer theme="colored" autoClose={2000} />
    </div>
};
