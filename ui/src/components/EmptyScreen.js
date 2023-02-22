import React from 'react'
import { Empty } from '../icons/Empty';

export const EmptyScreen = () => {
    return <>
        <div className="p-5">
            <div className="d-flex flex-column align-items-center">
                <Empty />
                <p className="mt-2">Submit your claim via freeform text</p>
                <p className="mb-0"><strong>Need to bulk upload?</strong></p>
                <p>Add a file (.csv) and select Analyse</p>
            </div>
        </div>
        <hr className="my-4" />
    </>
};
