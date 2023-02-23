import React, { useEffect, useState } from 'react';
import keywordExtractor from 'keyword-extractor';
import ReactWordcloud from 'react-wordcloud';
import { Modal, ModalHeader } from 'reactstrap';
import { ClaimsTable } from './ClaimsTable';
import Highlighter from 'react-highlight-words';

export const KeywordWordcloud = ({ data }) => {
    const [keywords, setKeywords] = useState([]);
    const [modalWord, setModalWord] = useState({});
    useEffect(() => {
        const contentWithKeywords = data.map(item => ({ 
            ...item, 
            keywords: keywordExtractor.extract(item.content, {
                language: "english",
                remove_duplicates: false
            }) 
        }));
        let weightedKeywords = [];
        for (const item of contentWithKeywords) {
            for (const word of item.keywords) {
                const wordIndex = weightedKeywords.findIndex(w => w.key === word);
                if (wordIndex !== -1) {
                    weightedKeywords[wordIndex].value = weightedKeywords[wordIndex].value + item.percentage
                    weightedKeywords[wordIndex].data.push(item)
                } else {
                    weightedKeywords.push({ key: word, value: item.percentage, data: [item] })
                }
            }
        }
        setKeywords(weightedKeywords.filter(a => ![
            '-', '~', '&amp'
        ].includes(a.key)).sort((a, b) => b.value - a.value).slice(0, 100))
    }, [data]);

    const options = {
        enableTooltip: false,
        fontFamily: 'nunito',
        fontSizes: [20, 75],
        fontStyle: 'normal',
        fontWeight: '800',
        rotations: 0,
        padding: 0,
        rotationAngles: [0, 90],
        scale: 'linear',
        spiral: 'rectangular',
        transitionDuration: 1000,
        deterministic: true,
    };

    return <div>
        <h3>Keywords</h3>
        <ReactWordcloud options={options}
            words={keywords.map(item => ({
                text: item.key,
                ...item
            }))}
            callbacks={{ onWordClick: setModalWord }}
        />
        <Modal isOpen={modalWord.text !== undefined} toggle={() => setModalWord({})} size="xl">
            <ModalHeader toggle={() => setModalWord({})}>
                {modalWord.text || ''}
            </ModalHeader>
            {modalWord.text && <ClaimsTable data={modalWord.data.map(item => ({
                ...item,
                content: <Highlighter searchWords={[modalWord.text]}
                    textToHighlight={item.content}
                    highlightClassName="p-0"
                />
            }))} keyword />}
        </Modal>
    </div>
};

