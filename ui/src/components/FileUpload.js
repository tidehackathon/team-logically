import React, { useState } from 'react'
import { Button, InputGroup, InputGroupText, Label } from 'reactstrap';
import { useCSVReader } from 'react-papaparse';

export const FileUpload = ({ onChange }) => {
    const { CSVReader } = useCSVReader();
    const [data, setData] = useState([]);

    return <div className="position-relative">
        <Label>File upload</Label>
        <CSVReader onUploadAccepted={(results) => setData(results.data)}
            config={{
                header: true,
                worker: true,
                skipEmptyLines: true,
                chunkSize: 1000000
            }}
        >
            {({
                getRootProps,
                acceptedFile,
                ProgressBar
            }) => (
                <>
                    <InputGroup>
                        <Button {...getRootProps()} color="primary">
                            File upload
                        </Button>
                        {acceptedFile && <InputGroupText>{acceptedFile.name}</InputGroupText>}
                        <Button color="primary"
                            disabled={!data.length}
                            onClick={() => onChange(data.map((item, i) => ({ ...item, id: acceptedFile.name + i })))}
                        >
                            Analyse
                        </Button>
                    </InputGroup>
                    <ProgressBar className="bg-danger position-absolute" style={{ bottom: -16 }} />
                </>
            )}
        </CSVReader>
    </div>
};
