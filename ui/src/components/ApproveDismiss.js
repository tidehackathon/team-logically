import React, { useState } from 'react'
import { toast } from 'react-toastify';
import { Button } from 'reactstrap';
import { Check } from 'react-feather';

export const ApproveDismiss = ({ handleDismiss }) => {
    const [approved, setApproved] = useState(false);
    return <div className="d-flex align-items-center">
        {approved ? <Check className="text-primary" /> : (
            <>
                <Button className="me-2" color="success" onClick={() => {
                    setApproved(true);
                    toast.success(`Accepted as disinformation`);
                }}>Approve</Button>
                <Button color="danger" onClick={handleDismiss}>Dismiss</Button>
            </>
        )}
    </div>
};
