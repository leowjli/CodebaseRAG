import { useState } from 'react';
import axios from '../api';

function Chat() {
    const [query, setQuery] = useState<string>('');
    const [response, setResponse] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);

    const handleQuery = async (): Promise<void> => {
        setLoading(true);
        try {
            const response = await axios.post('/query', { query });
            setResponse(response.data.message);
            setLoading(false);
        } catch(err) {
            console.error(err);
            setLoading(false);
        }
    };

    return (
        <div>
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Message Chat..."
            />
            <button onClick={handleQuery} disabled={loading}>
                {loading ? 'Processing...' : 'Send'}
            </button>
            {response && <p><strong>Answer:</strong>{response}</p>}
        </div>
    );
}

export default Chat;