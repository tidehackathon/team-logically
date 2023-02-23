import React, { useState } from 'react'
import { Button, Modal, ModalHeader } from 'reactstrap';
import { useAnalystOutcome } from '../AnalystOutcomeContext';
import { ClaimsTable } from './AnalyseData/ClaimsTable';

export const AnalystOutcomeModal = () => {
    const [modalOpen, setModalOpen] = useState(false);
    const { approved, dismissed } = useAnalystOutcome();
    return <div>
        <Button color="primary" outline onClick={() => setModalOpen(true)}>Analyst outcome</Button>
        <Modal isOpen={modalOpen} toggle={() => setModalOpen(false)} size="xl">
            <ModalHeader toggle={() => setModalOpen(false)}>
                Analyst outcome
            </ModalHeader>
            <ClaimsTable data={[...approved, ...dismissed].sort((a, b) => b.dateOfOutcome - a.dateOfOutcome)} outcome />
        </Modal>
    </div>
};
