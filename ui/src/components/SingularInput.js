import React, { useState } from 'react'
import { Button, Form, Input, InputGroup, Label } from 'reactstrap';

export const SingularInput = ({ onChange }) => {
    const [value, setValue] = useState('');
    return <Form onSubmit={(e) => { e.preventDefault(); onChange(value); }}>
        <Label>Singular</Label>
        <InputGroup>
            <Input type="text" value={value} onChange={(e) => setValue(e.target.value)} required />
            <Button color="primary" type="submit" disabled={!value.trim()}>Analyse</Button>
        </InputGroup>
    </Form>
};
