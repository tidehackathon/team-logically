import { useState } from "react";
import axios from 'axios';

export const useAPI = () => {
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState([]);
    const callAPI = (content) => {
        axios.post('http://localhost:6004/get_claims', { content }).then(res => {
            const keys = Object.keys(res.data.claim);

            setData(keys.map(key => ({
                id: key,
                claims: [res.data.claim[key]],
                percentage: res.data.disinfo_score[key],
                content: res.data.content[key],
            })));
            setLoading(false);
        }).catch(err => {
            console.error(err);
            setLoading(false);
        })
    }
    return {
        data, 
        loading,
        callAPI
    }
}
