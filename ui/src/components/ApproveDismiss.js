import React from 'react'
import { Button } from 'reactstrap';

export const ApproveDismiss = ({ handleDismiss }) => {
    return <div>
        <Button className="me-2" color="success">Approve</Button>
        <Button color="danger" onClick={handleDismiss}>Dismiss</Button>
    </div>
};
