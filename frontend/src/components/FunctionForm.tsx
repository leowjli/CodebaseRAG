import { useState } from 'react';
import axios from '../api';

interface FunctionFormProps {
    repoPath: string;
    setFunctions: (functions: []) => void;
}

function FunctionForm({ repoPath, setFunctions }: FunctionFormProps) {
    const [fileName, setFileName] = useState<string>('');
    const [lang, setLang] = useState<string>('python');
    const [loading, setLoading] = useState<boolean>(false);

    const handleExtractFunc = async (): Promise<void> => {
        setLoading(true);
        try {
            const endpoint = lang === 'python'
                ? '/extract-py-functions/'
                : '/extract-ts-functions/';
            const response = await axios.post(endpoint, {
                repoUrl: repoPath,
                file_name: fileName,
            });
            setFunctions(response.data.message);
            setLoading(false);
        } catch(err) {
            console.error(err);
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Extract Functions</h2>
            <input
                type="text"
                value={fileName}
                onChange={(e) => setFileName(e.target.value)}
                placeholder="Enter file name"
            />
            <select onChange={(e) => setLang(e.target.value)} value={lang}>
                <option value="python">Python</option>
                <option value="typescript">TypeScript</option>
            </select>
            <button onClick={handleExtractFunc} disabled={loading}>
                {loading ? 'Extracting...' : 'Extract Functions'}
            </button>
        </div>
    );
}

export default FunctionForm;