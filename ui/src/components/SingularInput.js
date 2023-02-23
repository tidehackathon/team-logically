import React, { useState } from 'react'
import { Button, Form, Input, InputGroup } from 'reactstrap';

export const SingularInput = ({ onChange }) => {
    const [value, setValue] = useState('');
    return <Form onSubmit={(e) => { e.preventDefault(); onChange(value); }}>
        <InputGroup>
            <Input type="text" placeholder="Enter singular claim" value={value} onChange={(e) => setValue(e.target.value)} required />
            <Button color="primary" type="submit" disabled={!value.trim()}>Analyse</Button>
        </InputGroup>
    </Form>
};
