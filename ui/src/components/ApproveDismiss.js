import React from 'react'
import { toast } from 'react-toastify';
import { Button } from 'reactstrap';
import { Check, X } from 'react-feather';
import moment from 'moment';
import { useAnalystOutcome } from '../AnalystOutcomeContext';

export const ApproveDismiss = ({ data }) => {
    const { approved, setApproved, dismissed, setDismissed } = useAnalystOutcome();
    if (dismissed.findIndex(item => item.id === data.id) !== -1) {
        return (
            <div className="d-flex align-items-center">
                <X className="text-danger mr-2" />
                Dismissed
            </div>
        );
    }
    if (approved.findIndex(item => item.id === data.id) !== -1) {
        return (
            <div className="d-flex align-items-center">
                <Check className="text-primary mr-2" />
                Approved
            </div>
        );
    }
    return <div className="d-flex align-items-center">
        <Button className="me-2" color="success" onClick={() => {
            setApproved([...approved, { ...data, dateOfOutcome: new Date() }]);
            toast(`Accepted as disinformation at ${moment().format('DD/MM/YYYY h:mm a')}`);
        }}>Approve</Button>
        <Button color="danger" onClick={() => {
            setDismissed([...dismissed, { ...data, dateOfOutcome: new Date() }]);
            toast(`Content dismissed at ${moment().format('DD/MM/YYYY h:mm a')}`);
        }}>Dismiss</Button>
    </div>
};
