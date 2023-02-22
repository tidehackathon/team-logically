import React, { useState } from 'react'
import { Button, Label } from 'reactstrap';
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
                    <div className="d-flex">
                        <Button {...getRootProps()} color="primary" className="me-2">
                            File upload
                        </Button>
                        {acceptedFile && <div className="border px-1 me-2 d-flex align-items-center">
                            <p className="m-0">{acceptedFile.name}</p>
                        </div>}
                        <Button color="primary" disabled={!data.length} onClick={() => onChange(data)}>Analyse</Button>
                    </div>
                    <ProgressBar className="bg-danger position-absolute" style={{ bottom: -16 }} />
                </>
            )}
        </CSVReader>
    </div>
};
